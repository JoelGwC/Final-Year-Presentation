import torch
import numpy as np
import joblib
from model import SurrogateModel

# =============================================================================
# RAFFC Three-Stage Op-Amp Circuit Evaluator
# =============================================================================
# Transistor Map (from schematic):
#
#   M0          — PMOS  — Tail current source (VDD-side). Gate = VB1.
#   M1, M2      — PMOS  — Differential input pair (gm1). Sources share M0.
#   M3          — NMOS  — Bottom sink for M1 branch only. Gate = VB2.
#   M4          — NMOS  — Bottom sink for M2 + RAFFC cascode branch. Gate = VB2.
#   M5, M6      — NMOS  — RAFFC cascode pair (gmb). Gate = VB3.
#   M7, M8      — PMOS  — Current mirror load for first stage. Gate = VB1.
#   M9          — PMOS  — Second stage inverting amplifier (gm2).
#   M10         — NMOS  — Sink for second stage. Gate = VB2.
#   M11, M13    — PMOS  — Third stage output driver (gm3). Both share same W/L.
#   M12, M14    — NMOS  — Sinks for third stage. Gate = VB2.
#   CC1         — Cap   — Primary Miller compensation cap (output node -> M9 gate).
#   CC2         — Cap   — Secondary compensation cap (M9 drain -> M5/M6 drain).
#   CL          — Cap   — Load capacitor at Vout.
#
# KCL Summary:
#   M0  carries : 2 * Id_1      (both diff pair arms)
#   M3  carries : Id_1          (M1 arm only)
#   M4  carries : Id_1 + Id_b   (M2 arm + RAFFC cascode arm)
#   M7/M8 carry : Id_b          (mirror load for RAFFC branch)
#   M9  carries : Id_2
#   M10 carries : Id_2
#   M11/M13 each carry: Id_3
#   M12/M14 each carry: Id_3
# =============================================================================


class RAFFC_OpAmp:
    def __init__(self, nmos_pth, pmos_pth,
                 scaler_X_nmos_path, scaler_y_nmos_path,
                 scaler_X_pmos_path, scaler_y_pmos_path):

        # Load scalers
        self.scaler_X_nmos = joblib.load(scaler_X_nmos_path)
        self.scaler_y_nmos = joblib.load(scaler_y_nmos_path)
        self.scaler_X_pmos = joblib.load(scaler_X_pmos_path)
        self.scaler_y_pmos = joblib.load(scaler_y_pmos_path)

        # Load surrogate ANN models
        self.nmos_model = SurrogateModel()
        self.nmos_model.load_state_dict(torch.load(nmos_pth))
        self.nmos_model.eval()

        self.pmos_model = SurrogateModel()
        self.pmos_model.load_state_dict(torch.load(pmos_pth))
        self.pmos_model.eval()

        # Global circuit specs
        self.VDD  = 1.0      # V
        self.VSS  = 0.0      # V
        self.VCM  = 0.5      # V  — common-mode input, locked
        self.CL   = 500e-12  # 500 pF load cap
        self.CC1  = 11e-12   # 11 pF primary Miller cap (fixed design parameter)
        self.GBW_target = 1e5  # 5 MHz target for CC1 consistency

        # Fixed bias network operating point (gm/Id=10, L=400nm)
        self.BIAS_GM_ID = 10.0
        self.BIAS_L     = 400e-9

    # -------------------------------------------------------------------------
    # ANN Query
    # -------------------------------------------------------------------------
    def get_transistor_params(self, gm_id, L, vds, is_nmos=True):
        """
        Query surrogate ANN.
        Returns: (id_w, gds_w, vgs, vdsat, cgg_w, cdd_w)
          id_w  [A/m]  — drain current density
          gds_w [S/m]  — output conductance density
          vgs   [V]    — gate-source voltage (signed: negative for PMOS)
          vdsat [V]    — saturation voltage (signed: negative for PMOS)
          cgg_w [F/m]  — total gate cap density
          cdd_w [F/m]  — drain cap density
        """
        raw_inputs = np.array([[gm_id, L, vds]])
        with torch.no_grad():
            if is_nmos:
                scaled_in  = self.scaler_X_nmos.transform(raw_inputs)
                tensor_in  = torch.tensor(scaled_in, dtype=torch.float32)
                scaled_out = self.nmos_model(tensor_in).numpy()
                real_out   = self.scaler_y_nmos.inverse_transform(scaled_out)
            else:
                scaled_in  = self.scaler_X_pmos.transform(raw_inputs)
                tensor_in  = torch.tensor(scaled_in, dtype=torch.float32)
                scaled_out = self.pmos_model(tensor_in).numpy()
                real_out   = self.scaler_y_pmos.inverse_transform(scaled_out)

        id_w, gds_w, vgs, vdsat, cgg_w, cdd_w = real_out[0]
        return id_w, gds_w, vgs, vdsat, cgg_w, cdd_w

    # -------------------------------------------------------------------------
    # Per-node VDS estimation (used to make ANN queries topology-aware)
    # -------------------------------------------------------------------------
    def _estimate_node_voltages(self, vgs_1, vdsat_L, vgs_2, vgs_3,
                                vgs_b, vdsat_m0, vgs_pmos_bias, vgs_nmos_bias):
        """
        Estimate all internal node voltages from a first KVL pass.
        All vgs/vdsat values are signed (negative for PMOS, positive for NMOS).

        Node definitions:
          V_tail  — common source of M1/M2 (PMOS diff pair), just below M0
          V_FS    — folded source node = drain of M3/M4 = source of M5/M6
          V_out1  — drain of M5/M6 = drain of M7/M8 = gate of M9
          V_out2  — drain of M9 = gate of M11/M13
          V_out   — output (target ~VCM = 0.5V in DC)
        """
        # M1/M2 are PMOS: source = V_tail, gate = VCM
        # VGS_pmos = VG - VS = VCM - V_tail  => V_tail = VCM - vgs_1
        # vgs_1 is negative for PMOS so -vgs_1 is positive
        V_tail = self.VCM - vgs_1          # VCM + |vgs_1|

        # M3/M4 are NMOS: source = VSS, V_FS = VSS + Vdsat_L + margin
        V_FS = self.VSS + abs(vdsat_L) + 0.05

        # M5/M6 are NMOS cascodes: source = V_FS, gate = VB3
        # V_out1 = drain of M7/M8 = drain of M5/M6
        # From M7/M8 (PMOS mirror): source = VDD, gate = VB1
        # VGS_m7 = VB1 - VDD = vgs_pmos_bias  (negative)
        # V_out1 = VDD + vgs_pmos_bias  (PMOS gate voltage sets drain operating point)
        # More precisely: V_out1 must leave M7/M8 in saturation.
        # Use: V_out1 = VDD - |Vdsat_m7| - margin (worst-case, conservative lower bound)
        # But we also need V_out1 > V_FS + |Vdsat_b| for M5/M6 saturation.
        # Best estimate: V_out1 = VDD + vgs_pmos_bias (the natural mirror operating point)
        V_out1 = self.VDD + vgs_pmos_bias  # VDD - |vgs_pmos_bias|

        # M9 is PMOS: gate = V_out1, source = VDD
        # V_out2 = VDD + vgs_2 = VDD - |vgs_2|
        V_out2 = self.VDD + vgs_2          # VDD - |vgs_2|

        # M11/M13 are PMOS: gate = V_out2, source = VDD
        # Output target ~0.5V. Drain of M11 = Vout.
        V_out = self.VCM                   # Target operating point

        return V_tail, V_FS, V_out1, V_out2, V_out

    # -------------------------------------------------------------------------
    # Main Evaluation
    # -------------------------------------------------------------------------
    def evaluate(self, sizing_guesses):
        """
        Evaluate RAFFC op-amp performance for a given set of sizing variables.

        sizing_guesses (14 values):
          [0]  gm_id_1  — gm/Id for M1/M2 (PMOS diff pair)
          [1]  L_1      — channel length for M1/M2
          [2]  gm_id_L  — gm/Id for M3/M4 (NMOS sinks)
          [3]  L_L      — channel length for M3/M4
          [4]  gm_id_2  — gm/Id for M9 (PMOS 2nd stage)
          [5]  L_2      — channel length for M9
          [6]  gm_id_3  — gm/Id for M11/M13 (PMOS 3rd stage)
          [7]  L_3      — channel length for M11/M13
          [8]  gm_id_b  — gm/Id for M5/M6 (NMOS RAFFC cascodes)
          [9]  L_b      — channel length for M5/M6
          [10] Id_1     — branch current through each M1, M2 arm
          [11] Id_2     — branch current through M9/M10
          [12] Id_3     — branch current through each M11/M13 arm
          [13] Id_b     — branch current through each M5/M6 arm

        Returns: (DC_Gain_dB, GBW_hz, PM_deg, Power, CC2_req)
        """
        INFEASIBLE = (-1.0, 0.0, -180.0, 1e9, 0.0)

        # ------------------------------------------------------------------
        # 1. EXTRACT OPTIMIZER VARIABLES
        # ------------------------------------------------------------------
        gm_id_1, L_1 = sizing_guesses[0],  sizing_guesses[1]   # M1/M2 PMOS diff pair
        gm_id_L, L_L = sizing_guesses[2],  sizing_guesses[3]   # M3/M4 NMOS sinks
        gm_id_2, L_2 = sizing_guesses[4],  sizing_guesses[5]   # M9    PMOS 2nd stage
        gm_id_3, L_3 = sizing_guesses[6],  sizing_guesses[7]   # M11/M13 PMOS 3rd stage
        gm_id_b, L_b = sizing_guesses[8],  sizing_guesses[9]   # M5/M6 NMOS RAFFC cascodes
        Id_1          = sizing_guesses[10]                       # Current per M1/M2 arm
        Id_2          = sizing_guesses[11]                       # Current through M9
        Id_3          = sizing_guesses[12]                       # Current per M11/M13 arm
        Id_b          = sizing_guesses[13]                       # Current per M5/M6 arm

        # ------------------------------------------------------------------
        # 2. FIRST-PASS ANN QUERY AT NOMINAL VDS = VDD/2
        #    Purpose: get vgs/vdsat to compute node voltages for the 2nd pass
        # ------------------------------------------------------------------
        vds_nom = self.VDD / 2.0

        # Query bias network PMOS/NMOS to get VGS for node voltage estimation
        _, _, vgs_pmos_bias, vdsat_m0, _, _ = self.get_transistor_params(
            self.BIAS_GM_ID, self.BIAS_L, vds_nom, is_nmos=False)
        _, _, vgs_nmos_bias, vdsat_m10, _, _ = self.get_transistor_params(
            self.BIAS_GM_ID, self.BIAS_L, vds_nom, is_nmos=True)

        # First-pass queries for topology transistors (to get vgs/vdsat for node voltages)
        _, _, vgs_1_nom,  vdsat_1_nom,  _, _ = self.get_transistor_params(gm_id_1, L_1, vds_nom, is_nmos=False)
        _, _, vgs_L_nom,  vdsat_L_nom,  _, _ = self.get_transistor_params(gm_id_L, L_L, vds_nom, is_nmos=True)
        _, _, vgs_2_nom,  vdsat_2_nom,  _, _ = self.get_transistor_params(gm_id_2, L_2, vds_nom, is_nmos=False)
        _, _, vgs_3_nom,  vdsat_3_nom,  _, _ = self.get_transistor_params(gm_id_3, L_3, vds_nom, is_nmos=False)
        _, _, vgs_b_nom,  vdsat_b_nom,  _, _ = self.get_transistor_params(gm_id_b, L_b, vds_nom, is_nmos=True)

        # ------------------------------------------------------------------
        # 3. KVL NODE VOLTAGE ESTIMATION
        # ------------------------------------------------------------------
        V_tail, V_FS, V_out1, V_out2, V_out = self._estimate_node_voltages(
            vgs_1_nom, vdsat_L_nom, vgs_2_nom, vgs_3_nom,
            vgs_b_nom, vdsat_m0, vgs_pmos_bias, vgs_nmos_bias)

        # Clamp node voltages to physical rail limits (safety for ANN input)
        V_tail = np.clip(V_tail, self.VSS + 0.05, self.VDD - 0.05)
        V_FS   = np.clip(V_FS,   self.VSS + 0.05, V_tail - 0.05)
        V_out1 = np.clip(V_out1, V_FS   + 0.05, self.VDD - 0.05)
        V_out2 = np.clip(V_out2, self.VSS + 0.05, self.VDD - 0.05)

        # Per-transistor VDS estimates (all positive magnitudes for ANN input)
        # PMOS VDS = VS - VD  (source at high voltage)
        # NMOS VDS = VD - VS  (source at low voltage)
        vds_M1  = abs(V_tail - V_FS)    # M1/M2: source=V_tail, drain=V_FS (folded)
        vds_M3  = abs(V_FS   - self.VSS) # M3: drain=V_FS, source=VSS  — FIXED: M3 alone
        vds_M4  = abs(V_FS   - self.VSS) # M4: same structure, different current
        vds_M9  = abs(self.VDD - V_out2) # M9: source=VDD, drain=V_out2
        vds_M11 = abs(self.VDD - V_out)  # M11/M13: source=VDD, drain=Vout
        vds_M5  = abs(V_out1 - V_FS)     # M5/M6: drain=V_out1, source=V_FS

        # ------------------------------------------------------------------
        # 4. SECOND-PASS ANN QUERY AT TOPOLOGY-CORRECT VDS PER TRANSISTOR
        #    This is the key fix: each device sees its own realistic VDS.
        # ------------------------------------------------------------------
        id_w_1, gds_w_1, vgs_1, vdsat_1, cgg_w_1, cdd_w_1 = self.get_transistor_params(
            gm_id_1, L_1, vds_M1,  is_nmos=False)  # M1/M2 PMOS

        id_w_L, gds_w_L, vgs_L, vdsat_L, cgg_w_L, cdd_w_L = self.get_transistor_params(
            gm_id_L, L_L, vds_M3,  is_nmos=True)   # M3/M4 NMOS sinks

        id_w_2, gds_w_2, vgs_2, vdsat_2, cgg_w_2, cdd_w_2 = self.get_transistor_params(
            gm_id_2, L_2, vds_M9,  is_nmos=False)  # M9 PMOS

        id_w_3, gds_w_3, vgs_3, vdsat_3, cgg_w_3, cdd_w_3 = self.get_transistor_params(
            gm_id_3, L_3, vds_M11, is_nmos=False)  # M11/M13 PMOS

        id_w_b, gds_w_b, vgs_b, vdsat_b, cgg_w_b, cdd_w_b = self.get_transistor_params(
            gm_id_b, L_b, vds_M5,  is_nmos=True)   # M5/M6 NMOS

        # Re-query bias network at their true VDS for accurate vdsat margin checks
        _, _, vgs_pmos_bias, vdsat_m0, _, _ = self.get_transistor_params(
            self.BIAS_GM_ID, self.BIAS_L,
            abs(self.VDD - V_tail), is_nmos=False)  # M0: VDS = VDD - V_tail
        _, _, vgs_nmos_bias, vdsat_m10, _, _ = self.get_transistor_params(
            self.BIAS_GM_ID, self.BIAS_L,
            abs(V_out2 - self.VSS), is_nmos=True)   # M10: VDS = V_out2 - VSS
        vdsat_m7 = vdsat_m0  # M7/M8 use same PMOS sizing as M0

        # ------------------------------------------------------------------
        # 5. GUARD: Check ANN outputs are physically plausible
        # ------------------------------------------------------------------
        id_w_1 = max(id_w_1, 1e-6)
        id_w_L = max(id_w_L, 1e-6)
        id_w_2 = max(id_w_2, 1e-6)
        id_w_3 = max(id_w_3, 1e-6)
        id_w_b = max(id_w_b, 1e-6)

        # ------------------------------------------------------------------
        # 6. PHYSICAL WIDTHS
        #
        #   Folded cascode KCL at M3 drain node:
        #     M1 injects Id_1 downward (PMOS diff pair)
        #     M7 injects Id_b downward (PMOS current mirror)
        #     M5 source connects here; M3 sinks the sum: Id_1 + Id_b
        #
        #   Same argument applies symmetrically to M4:
        #     M2 injects Id_1, M8 injects Id_b → M4 sinks Id_1 + Id_b
        #
        #   M3 and M4 are matched devices carrying the same current.
        # ------------------------------------------------------------------
        W_1       = Id_1          / id_w_1   # M1 = M2 (matched diff pair)
        W_M3_M4   = (Id_1 + Id_b) / id_w_L   # M3 = M4 (matched folded sinks)
        W_2       = Id_2          / id_w_2   # M9
        W_3       = Id_3          / id_w_3   # M11 = M13 (matched)
        W_b       = Id_b          / id_w_b   # M5 = M6 (matched)

        # Minimum width enforcement (PDK floor = 120 nm)
        MIN_W = 120e-9
        if any(w < MIN_W for w in [W_1, W_M3_M4, W_2, W_3, W_b]):
            return INFEASIBLE

        # ------------------------------------------------------------------
        # 7. TRANSCONDUCTANCES AND OUTPUT RESISTANCES
        # ------------------------------------------------------------------
        gm1  = gm_id_1 * Id_1
        gm2  = gm_id_2 * Id_2
        gm3  = gm_id_3 * Id_3
        gmb  = gm_id_b * Id_b

        # Asymptotic stability condition: gmb must dominate gm1
        if gm1 >= gmb:
            return INFEASIBLE

        # Stage output resistances: ro = 1 / (W * gds_w)
        ro1 = 1.0 / max(W_1  * gds_w_1, 1e-12)   # First stage: ro_M1 || ro_M3
        ro2 = 1.0 / max(W_2  * gds_w_2, 1e-12)   # Second stage: ro_M9 || ro_M10
        ro3 = 1.0 / max(W_3  * gds_w_3, 1e-12)   # Third stage: ro_M11 || ro_M12

        # ------------------------------------------------------------------
        # 8. SATURATION HEADROOM CHECKS (KVL-consistent, topology-accurate)
        #
        #   Rule: VDS_actual >= |Vdsat| + margin  =>  margin_value >= 0
        #   margin_value < 0 means the device is heading into triode.
        #
        #   All vdsat values are already topology-correct from the 2nd ANN pass.
        # ------------------------------------------------------------------
        margins = {
            "M0  (Tail PMOS)":    (self.VDD - V_tail) - abs(vdsat_m0)  - 0.10,
            "M1/M2 (Diff Pair)":  (V_tail   - V_FS)   - abs(vdsat_1)   - 0.05,
            "M3  (Sink M1 arm)":  (V_FS     - self.VSS)- abs(vdsat_L)   - 0.05,
            "M4  (Sink M2 arm)":  (V_FS     - self.VSS)- abs(vdsat_L)   - 0.05,
            "M5/M6 (RAFFC)":      (V_out1   - V_FS)   - abs(vdsat_b)   - 0.05,
            "M7/M8 (Mirror)":     (self.VDD - V_out1)  - abs(vdsat_m7)  - 0.05,
            "M9  (2nd Stage)":    (self.VDD - V_out2)  - abs(vdsat_2)   - 0.05,
            "M10 (2nd Sink)":     (V_out2   - self.VSS)- abs(vdsat_m10) - 0.10,
            "M11/M13 (3rd Stg)":  (self.VDD - V_out)   - abs(vdsat_3)   - 0.05,
        }

        sat_penalty = 0.0
        violated = []
        for name, margin in margins.items():
            if margin < 0.0:
                sat_penalty += (abs(margin) ** 2) * 1e8
                violated.append((name, margin))

        # If ANY transistor violates saturation, the design is physically dead.
        # Return infeasible immediately so the optimizer never treats this as
        # a valid (Gain, GBW, PM) triple — prevents constraint satisfaction
        # masking by a penalty-inflated Power term.
        if sat_penalty > 0.0:
            # Return with the sat_penalty encoded in Power so NSGA-II can
            # still use gradient information to move away from this region,
            # but Gain/PM are clearly infeasible.
            return INFEASIBLE[0], INFEASIBLE[1], INFEASIBLE[2], 1e9 + sat_penalty, 0.0

        # ------------------------------------------------------------------
        # 9. PERFORMANCE METRICS
        # ------------------------------------------------------------------

        # DC Gain
        # First-stage gain: gm1 drives ro1. Due to folded topology the
        # effective load is ro_M1 || ro_M3, but here ro1 already represents
        # the dominant output resistance at the first-stage output node.
        Av_linear = (gm1 * ro1) * (gm2 * ro2) * (gm3 * ro3)
        if Av_linear <= 0.0 or not np.isfinite(Av_linear):
            return INFEASIBLE
        DC_Gain_dB = 20.0 * np.log10(Av_linear)

        # GBW — dominated by CC1 at first stage output (standard Miller)
        GBW_hz = gm1 / (2.0 * np.pi * self.CC1)

        # Phase Margin — RAFFC analytical model
        # Ratio κ = gmb/gm1 determines the intrinsic PM of the RAFFC topology.
        # Reference: RAFFC paper equations for asymptotic PM.
        kappa = gmb / max(gm1, 1e-9)
        ideal_pm_rad = np.arctan((kappa ** 3) / (kappa ** 2 + 2.0))
        ideal_pm_deg = np.degrees(ideal_pm_rad)

        # Parasitic pole due to M9 gate capacitance at the folded output node
        C_gate_M9   = W_2 * cgg_w_2
        omega_p_par = gmb / max(C_gate_M9, 1e-15)
        parasitic_delay_deg = np.degrees(
            np.arctan((2.0 * np.pi * GBW_hz) / omega_p_par))

        PM_deg = ideal_pm_deg - parasitic_delay_deg

        # CC2 requirement (from RAFFC paper: CC2 ensures LHP zero placement)
        CC2_req = (2.0 * gm3 * (self.CC1 ** 2)) / (gmb * self.CL)

        # ------------------------------------------------------------------
        # 10. POWER (no penalty — only feasible designs reach here)
        #
        #   Total supply current:
        #     M0 branch : 2 * Id_1         (feeds both diff pair arms)
        #     M9 branch : Id_2
        #     M11 branch: 2 * Id_3         (two output driver transistors)
        #     M5/M6 arm : 2 * Id_b         (two RAFFC cascodes)
        # ------------------------------------------------------------------
        I_total = (2.0 * Id_1) + Id_2 + (2.0 * Id_3) + (2.0 * Id_b)
        Power   = self.VDD * I_total

        if not np.isfinite(DC_Gain_dB) or not np.isfinite(GBW_hz) or not np.isfinite(PM_deg):
            return INFEASIBLE

        return DC_Gain_dB, GBW_hz, PM_deg, Power, CC2_req

    # -------------------------------------------------------------------------
    # Physical Dimensions Blueprint (post-optimization)
    # -------------------------------------------------------------------------
    def calculate_physical_dimensions(self, final_optimal_guesses, bias_currents):
        """
        Compute and print the complete physical blueprint for Cadence entry.
        Uses the same two-pass VDS methodology as evaluate() for consistency.

        Parameters
        ----------
        final_optimal_guesses : array-like, length 10
            [gm_id_1, L_1, gm_id_L, L_L, gm_id_2, L_2,
             gm_id_3, L_3, gm_id_b, L_b]
        bias_currents : array-like, length 4
            [Id_1, Id_2, Id_3, Id_b]

        Returns
        -------
        (W_1, W_M3_M4, W_2, W_3, W_b) — widths in metres
        """
        gm_id_1, L_1 = final_optimal_guesses[0], final_optimal_guesses[1]
        gm_id_L, L_L = final_optimal_guesses[2], final_optimal_guesses[3]
        gm_id_2, L_2 = final_optimal_guesses[4], final_optimal_guesses[5]
        gm_id_3, L_3 = final_optimal_guesses[6], final_optimal_guesses[7]
        gm_id_b, L_b = final_optimal_guesses[8], final_optimal_guesses[9]
        Id_1, Id_2, Id_3, Id_b = bias_currents

        vds_nom = self.VDD / 2.0

        # ---- Pass 1: nominal VDS to get node voltages ----
        _, _, vgs_pmos_bias, vdsat_m0, _, _ = self.get_transistor_params(
            self.BIAS_GM_ID, self.BIAS_L, vds_nom, is_nmos=False)
        _, _, vgs_nmos_bias, vdsat_m10, _, _ = self.get_transistor_params(
            self.BIAS_GM_ID, self.BIAS_L, vds_nom, is_nmos=True)

        _, _, vgs_1_nom, vdsat_L_nom, _, _ = self.get_transistor_params(gm_id_1, L_1, vds_nom, is_nmos=False)
        _, _, vgs_L_nom, _,           _, _ = self.get_transistor_params(gm_id_L, L_L, vds_nom, is_nmos=True)
        _, _, vgs_2_nom, _,           _, _ = self.get_transistor_params(gm_id_2, L_2, vds_nom, is_nmos=False)
        _, _, vgs_3_nom, _,           _, _ = self.get_transistor_params(gm_id_3, L_3, vds_nom, is_nmos=False)
        _, _, vgs_b_nom, _,           _, _ = self.get_transistor_params(gm_id_b, L_b, vds_nom, is_nmos=True)

        V_tail, V_FS, V_out1, V_out2, V_out = self._estimate_node_voltages(
            vgs_1_nom, vdsat_L_nom, vgs_2_nom, vgs_3_nom,
            vgs_b_nom, vdsat_m0, vgs_pmos_bias, vgs_nmos_bias)

        V_tail = np.clip(V_tail, self.VSS + 0.05, self.VDD - 0.05)
        V_FS   = np.clip(V_FS,   self.VSS + 0.05, V_tail - 0.05)
        V_out1 = np.clip(V_out1, V_FS   + 0.05, self.VDD - 0.05)
        V_out2 = np.clip(V_out2, self.VSS + 0.05, self.VDD - 0.05)

        vds_M1  = abs(V_tail - V_FS)
        vds_M3  = abs(V_FS   - self.VSS)
        vds_M9  = abs(self.VDD - V_out2)
        vds_M11 = abs(self.VDD - V_out)
        vds_M5  = abs(V_out1 - V_FS)

        # ---- Pass 2: topology-correct VDS ----
        id_w_1, _, vgs_1, _, _, _ = self.get_transistor_params(gm_id_1, L_1, vds_M1,  is_nmos=False)
        id_w_L, _, vgs_L, _, _, _ = self.get_transistor_params(gm_id_L, L_L, vds_M3,  is_nmos=True)
        id_w_2, _, vgs_2, _, _, _ = self.get_transistor_params(gm_id_2, L_2, vds_M9,  is_nmos=False)
        id_w_3, _, vgs_3, _, _, _ = self.get_transistor_params(gm_id_3, L_3, vds_M11, is_nmos=False)
        id_w_b, _, vgs_b, _, _, _ = self.get_transistor_params(gm_id_b, L_b, vds_M5,  is_nmos=True)

        id_w_pmos_bias, _, vgs_pmos_bias, _, _, _ = self.get_transistor_params(
            self.BIAS_GM_ID, self.BIAS_L, abs(self.VDD - V_tail), is_nmos=False)
        id_w_nmos_bias, _, vgs_nmos_bias, _, _, _ = self.get_transistor_params(
            self.BIAS_GM_ID, self.BIAS_L, abs(V_out2 - self.VSS),  is_nmos=True)

        id_w_1 = max(id_w_1, 1e-6)
        id_w_L = max(id_w_L, 1e-6)
        id_w_2 = max(id_w_2, 1e-6)
        id_w_3 = max(id_w_3, 1e-6)
        id_w_b = max(id_w_b, 1e-6)
        id_w_pmos_bias = max(id_w_pmos_bias, 1e-6)
        id_w_nmos_bias = max(id_w_nmos_bias, 1e-6)

        # ---- Widths — M3 and M4 are matched, both carry Id_1 + Id_b ----
        # KCL at M3 drain node: M1 injects Id_1, M7 injects Id_b → M3 sinks sum
        # KCL at M4 drain node: M2 injects Id_1, M8 injects Id_b → M4 sinks sum
        W_1      = Id_1          / id_w_1
        W_M3_M4  = (Id_1 + Id_b) / id_w_L   # M3 = M4 (matched folded sinks)
        W_2      = Id_2          / id_w_2
        W_3      = Id_3          / id_w_3    # M11 = M13
        W_b      = Id_b          / id_w_b    # M5 = M6

        # ---- Biasing network widths (KCL-aligned) ----
        W_M0       = (2.0 * Id_1) / id_w_pmos_bias   # M0: total tail = 2*Id_1
        W_M7 = W_M8 = Id_b        / id_w_pmos_bias   # M7/M8: mirror for RAFFC branch
        W_M10      = Id_2          / id_w_nmos_bias   # M10: 2nd stage sink
        W_M12 = W_M14 = Id_3      / id_w_nmos_bias   # M12/M14: 3rd stage sinks

        # ---- Bias voltages ----
        # VB1: sets gate of M0, M7, M8 (PMOS) — one Vgs below VDD
        vgs_pmos_clean = max(min(abs(vgs_pmos_bias), 0.55), 0.30)  # clamp to safe range
        VB1 = self.VDD - vgs_pmos_clean

        # VB2: sets gate of M3, M4, M10, M12, M14 (NMOS) — one Vgs above VSS
        VB2 = abs(vgs_nmos_bias)

        # VB3: sets gate of M5/M6 (NMOS cascodes) — needs enough headroom
        # VB3 = VGS_nmos + Vdsat_M3 so that M3/M4 just remain in saturation
        vds_sat_M3 = 2.0 / gm_id_L   # Approximate: Vdsat ~ 2/(gm/Id)
        VB3 = abs(vgs_nmos_bias) + abs(vds_sat_M3)

        # ---- Capacitors — use self.CC1 consistently (no re-derivation) ----
        gm1     = gm_id_1 * Id_1
        gm3     = gm_id_3 * Id_3
        gmb     = gm_id_b * Id_b
        CC2_val = (2.0 * gm3 * (self.CC1 ** 2)) / (gmb * self.CL)

        # ---- Print Blueprint ----
        sep = "=" * 50
        print(f"\n{sep}")
        print("  RAFFC OP-AMP — CADENCE BLUEPRINT")
        print(f"{sep}")

        print("\n--- CORE AMPLIFIER TRANSISTORS ---")
        print(f"  M1, M2  (PMOS Diff Pair)    W={W_1*1e6:.3f} um   L={L_1*1e9:.0f} nm   Id={Id_1*1e6:.2f} uA each")
        print(f"  M3, M4  (NMOS Folded Sinks) W={W_M3_M4*1e6:.3f} um  L={L_L*1e9:.0f} nm   Id={(Id_1+Id_b)*1e6:.2f} uA each")
        print(f"  M5, M6  (NMOS RAFFC Casc.)  W={W_b*1e6:.3f} um   L={L_b*1e9:.0f} nm   Id={Id_b*1e6:.2f} uA each")
        print(f"  M9      (PMOS 2nd Stage)    W={W_2*1e6:.3f} um   L={L_2*1e9:.0f} nm   Id={Id_2*1e6:.2f} uA")
        print(f"  M11,M13 (PMOS 3rd Stage)    W={W_3*1e6:.3f} um   L={L_3*1e9:.0f} nm   Id={Id_3*1e6:.2f} uA each")

        print("\n--- BIASING NETWORK  (gm/Id=10, L=400nm) ---")
        print(f"  M0      (PMOS Tail Mirror)  W={W_M0*1e6:.3f} um   L=400 nm   Id={2*Id_1*1e6:.2f} uA")
        print(f"  M7, M8  (PMOS RAFFC Mirror) W={W_M7*1e6:.3f} um   L=400 nm   Id={Id_b*1e6:.2f} uA each")
        print(f"  M10     (NMOS 2nd Stg Sink) W={W_M10*1e6:.3f} um  L=400 nm   Id={Id_2*1e6:.2f} uA")
        print(f"  M12,M14 (NMOS 3rd Stg Sink) W={W_M12*1e6:.3f} um  L=400 nm   Id={Id_3*1e6:.2f} uA each")

        print("\n--- DC BIAS VOLTAGES ---")
        print(f"  VB1 = {VB1:.4f} V   (Gate of M0, M7, M8 — PMOS tail/mirror)")
        print(f"  VB2 = {VB2:.4f} V   (Gate of M3, M4, M10, M12, M14 — NMOS sinks)")
        print(f"  VB3 = {VB3:.4f} V   (Gate of M5, M6 — NMOS RAFFC cascodes)")

        print("\n--- INTERNAL NODE VOLTAGES (expected DC) ---")
        print(f"  V_tail  = {V_tail:.4f} V")
        print(f"  V_FS    = {V_FS:.4f} V   (folded source / M3-M4 drain)")
        print(f"  V_out1  = {V_out1:.4f} V  (1st stage output / M9 gate)")
        print(f"  V_out2  = {V_out2:.4f} V  (2nd stage output / M11 gate)")
        print(f"  V_out   = {V_out:.4f} V   (output, target = VCM)")

        print("\n--- PASSIVE COMPONENTS ---")
        print(f"  CC1 (Primary Miller)   = {self.CC1*1e12:.2f} pF")
        print(f"  CC2 (Secondary Comp.)  = {CC2_val*1e15:.2f} fF")
        print(f"  CL  (Load Cap)         = {self.CL*1e12:.0f} pF")

        print(f"\n--- PERFORMANCE ESTIMATES ---")
        print(f"  gm1  = {gm1*1e3:.3f} mA/V")
        print(f"  gm2  = {gm_id_2*Id_2*1e3:.3f} mA/V")
        print(f"  gm3  = {gm3*1e3:.3f} mA/V")
        print(f"  gmb  = {gmb*1e3:.3f} mA/V   (κ = gmb/gm1 = {gmb/max(gm1,1e-9):.2f})")
        I_total = (2.0*Id_1) + Id_2 + (2.0*Id_3) + (2.0*Id_b)
        print(f"  GBW  = {gm1/(2*np.pi*self.CC1)/1e6:.2f} MHz")
        print(f"  Power = {self.VDD*I_total*1e6:.2f} uW")
        print(f"{sep}\n")

        return W_1, W_M3_M4, W_2, W_3, W_b


# =============================================================================
# Quick sanity test
# =============================================================================
if __name__ == "__main__":
    circuit = RAFFC_OpAmp(
        'nmos_surrogate_model.pth', 'pmos_surrogate_model.pth',
        'scaler_X_nmos.pkl', 'scaler_y_nmos.pkl',
        'scaler_X_pmos.pkl', 'scaler_y_pmos.pkl')

    preds = circuit.get_transistor_params(gm_id=10.0, L=400e-9, vds=0.5, is_nmos=True)
    print("NMOS @ gm/Id=10, L=400nm, Vds=0.5V:")
    print(f"  Id/W={preds[0]*1e3:.3f} mA/um  gds/W={preds[1]*1e3:.3f} mS/um  "
          f"VGS={preds[2]:.3f}V  Vdsat={preds[3]:.3f}V")

    preds = circuit.get_transistor_params(gm_id=10.0, L=400e-9, vds=0.5, is_nmos=False)
    print("PMOS @ gm/Id=10, L=400nm, Vds=0.5V:")
    print(f"  Id/W={preds[0]*1e3:.3f} mA/um  gds/W={preds[1]*1e3:.3f} mS/um  "
          f"VGS={preds[2]:.3f}V  Vdsat={preds[3]:.3f}V")