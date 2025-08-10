import numpy as np
from scipy.special import j1
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

class FlexibleCylinderModel:
    """
    Flexible cylinder form factor model for SAXS fitting
    Based on Pedersen & Schurtenberger (1996) worm-like chain model
    Implements the exact C code logic from SasView
    """
    
    def __init__(self):
        """Initialize the model"""
        pass
    
    def sas_2J1x_x(self, x):
        """2*J1(x)/x cylinder form factor"""
        if np.isscalar(x):
            return 2.0 * j1(x) / x if x != 0.0 else 1.0
        else:
            x = np.asarray(x)
            return np.where(x != 0.0, 2.0 * j1(x) / x, 1.0)
    
    def Rgsquare(self, L, b):
        """
        Radius of gyration squared for flexible chain
        Pedersen eq 15 with excluded volume effects
        """
        x = L / b
        # Use Horner's method as in C code:
        # alpha^2 = [1.0 + (x/3.12)^2 + (x/8.67)^3] ^ (0.176/3)
        alphasq = (1.0 + x*x*(1.027284681130835e-01 + 1.534414548417740e-03*x))**(5.866666666666667e-02)
        return alphasq * L * b / 6.0
    
    def Rgsquareshort(self, L, b):
        """Modified Rg for short chains"""
        r = b / L  # = 1/n_b in Pedersen ref.
        rg_sq = self.Rgsquare(L, b)
        correction = 1.0 + r*(-1.5 + r*(1.5 + r*0.75*np.expm1(-2.0/r)))
        return rg_sq * correction
    
    def w_WR(self, x):
        """
        Weighting function from Pedersen eq. 16
        w = [1 + tanh((x-C4)/C5)]/2
        """
        C4 = 1.523
        C5 = 0.1477
        return 0.5 + 0.5*np.tanh((x - C4)/C5)
    
    def Sdebye(self, qsq):
        """Debye function with series expansion for small arguments"""
        DEBYE_CUTOFF = 0.25  # for double precision
        
        if np.isscalar(qsq):
            if qsq < DEBYE_CUTOFF:
                x = qsq
                # Pade approximant from mathematica
                A1, A2, A3, A4 = 1./15., 1./60, 0., 1./75600.
                B1, B2, B3, B4 = 2./5., 1./15., 1./180., 1./5040.
                return (((A4*x + A3)*x + A2)*x + A1)*x + 1. / ((((B4*x + B3)*x + B2)*x + B1)*x + 1.)
            else:
                return 2.*(np.expm1(-qsq) + qsq)/(qsq*qsq)
        else:
            # Vectorized version
            result = np.zeros_like(qsq)
            small_mask = qsq < DEBYE_CUTOFF
            large_mask = ~small_mask
            
            # Small q - only process if there are small values
            if np.any(small_mask):
                x = qsq[small_mask]
                A1, A2, A3, A4 = 1./15., 1./60, 0., 1./75600.
                B1, B2, B3, B4 = 2./5., 1./15., 1./180., 1./5040.
                result[small_mask] = (((A4*x + A3)*x + A2)*x + A1)*x + 1. / ((((B4*x + B3)*x + B2)*x + B1)*x + 1.)
            
            # Large q - only process if there are large values
            if np.any(large_mask):
                qsq_large = qsq[large_mask]
                result[large_mask] = 2.*(np.expm1(-qsq_large) + qsq_large)/(qsq_large*qsq_large)
            
            return result
    
    def Sexv(self, q, L, b):
        """
        Excluded volume structure factor (Pedersen eq 13, corrected by Chen)
        """
        C1 = 1.22
        C2 = 0.4288
        C3 = -1.651
        miu = 0.585
        
        qr = q * np.sqrt(self.Rgsquare(L, b))
        qr_miu = qr**(-1.0/miu)
        w = self.w_WR(qr)
        t10 = self.Sdebye(qr*qr) * (1.0 - w)
        t11 = ((C3*qr_miu + C2)*qr_miu + C1)*qr_miu
        
        return t10 + w*t11
    
    def Sexv_new(self, q, L, b):
        """Modified excluded volume function with correction term"""
        qr = q * np.sqrt(self.Rgsquare(L, b))
        qr2 = qr * qr
        C = 3.06 * (L/b)**(-0.44) if L/b > 10.0 else 1.0
        
        # Correction term t9
        t9 = C * b/L * (4.0 - np.exp(-qr2) * (11.0 + 7.0/qr2) + 7.0/qr2) / 15.0
        
        Sexv_orig = self.Sexv(q, L, b)
        
        # Calculate derivative to decide on correction
        del_factor = 1.05
        qdel = (self.Sexv(q*del_factor, L, b) - Sexv_orig) / (q*(del_factor - 1.0))
        
        if qdel < 0:
            return t9 + Sexv_orig
        else:
            w = self.w_WR(qr)
            t10 = self.Sdebye(qr2) * (1.0 - w)
            return t9 + t10
    
    def a_long(self, q, L, b):
        """Structure factor for long chains (L > 4b, q*b > 3.1)"""
        p1 = 4.12
        p2 = 4.42
        q0 = 3.1
        
        # Constants from least squares fit
        C1 = 1.22
        C2 = 0.4288
        C3 = -1.651
        C4 = 1.523
        C5 = 0.1477
        miu = 0.585
        
        C = 3.06 * (L/b)**(-0.44) if L/b > 10.0 else 1.0
        r2 = self.Rgsquare(L, b)
        r = np.sqrt(r2)
        qr_b = q0 * r / b
        qr_b_sq = qr_b * qr_b
        qr_b_4 = qr_b_sq * qr_b_sq
        qr_b_miu = qr_b**(-1.0/miu)
        em1_qr_b_sq = np.expm1(-qr_b_sq)
        sech2 = 1.0 / np.cosh((qr_b - C4)/C5)**2
        w = self.w_WR(qr_b)
        
        t1 = q0**(1.0 + p1 + p2) / (b * (p1 - p2))
        t2 = C/(15.0*L) * (14.0*b*b*em1_qr_b_sq/(q0*qr_b_sq) + 
                           2.0*q0*r2*np.exp(-qr_b_sq)*(11.0 + 7.0/qr_b_sq))
        
        t11 = ((C3*qr_b_miu + C2)*qr_b_miu + C1)*qr_b_miu
        t3 = r*sech2/(2.*C5)*t11
        t4 = r*(em1_qr_b_sq + qr_b_sq)*sech2 / (C5*qr_b_4)
        t5 = -4.0*r*qr_b*em1_qr_b_sq/qr_b_4 * (1.0 - w)
        t10 = 2.0*(em1_qr_b_sq + qr_b_sq)/qr_b_4 * (1.0 - w)
        t6 = 4.0*b/q0 * t10
        t7 = r*((-3.0*C3*qr_b_miu - 2.0*C2)*qr_b_miu - 1.0*C1)*qr_b_miu/(miu*qr_b)
        t9 = C*b/L * (4.0 - np.exp(-qr_b_sq) * (11.0 + 7.0/qr_b_sq) + 7.0/qr_b_sq)/15.0
        
        t12 = b*b*np.pi/(L*q0*q0) + t2 + t3 - t4 + t5 - t6 + t7*w
        t13 = -b*np.pi/(L*q0) + t9 + t10 + t11*w
        
        a1 = q0**p1 * t13 - t1 * q0**(-p2) * (t12 + b*p1/q0*t13)
        a2 = t1 * q0**(-p1) * (t12 + b*p1/q0*t13)
        
        ans = a1 * (q*b)**(-p1) + a2 * (q*b)**(-p2) + np.pi/(q*L)
        return ans
    
    def _short_helper(self, r2, exp_qr_b, L, b, p1short, p2short, q0):
        """Helper function for short chain calculation"""
        qr2 = q0*q0 * r2
        b3 = b*b*b
        q0p = q0**(-4.0 + p1short)
        
        yy = (1.0/(L*r2*r2) * b/exp_qr_b * q0p * 
              (8.0*b3*L - 8.0*b3*exp_qr_b*L + 2.0*b3*exp_qr_b*L*p2short - 
               2.0*b*exp_qr_b*L*p2short*qr2 + 4.0*b*exp_qr_b*L*qr2 - 
               2.0*b3*L*p2short + 4.0*b*L*qr2 - np.pi*exp_qr_b*qr2*q0*r2 + 
               np.pi*exp_qr_b*p2short*qr2*q0*r2))
        
        return yy
    
    def a_short(self, qp, L, b, q0):
        """Structure factor for short chains (L <= 4b, q*b > q0short)"""
        p1short = 5.36
        p2short = 5.62
        
        r2 = self.Rgsquareshort(L, b)
        exp_qr_b = np.exp(r2 * (q0/b)**2)
        pdiff = p1short - p2short
        
        a1 = self._short_helper(r2, exp_qr_b, L, b, p1short, p2short, q0) / pdiff
        a2 = -self._short_helper(r2, exp_qr_b, L, b, p2short, p1short, q0) / pdiff
        
        ans = a1 * (qp*b)**(-p1short) + a2 * (qp*b)**(-p2short) + np.pi/(qp*L)
        return ans
    
    def Sk_WR(self, q, L, b):
        """
        Main worm-like chain structure factor function
        """
        Rg_short = np.sqrt(self.Rgsquareshort(L, b))
        q0short = max(1.9/Rg_short, 3.0)
        
        if L > 4*b:  # Longer chains
            if q*b <= 3.1:
                ans = self.Sexv_new(q, L, b)
            else:  # q*b > 3.1
                ans = self.a_long(q, L, b)
        else:  # Shorter chains (L <= 4*b)
            if q*b <= q0short:  # q*b <= max(1.9/Rg_short, 3)
                ans = self.Sdebye((q*Rg_short)**2)
            else:  # q*b > max(1.9/Rg_short, 3)
                ans = self.a_short(q, L, b, q0short)
        
        return ans
    
    def gauss_coil(self, qm):
        """
        Gaussian coil form factor from SasView C implementation
        
        Parameters:
        -----------
        qm : float or array_like
            q * Rg_mol (dimensionless parameter)

        Returns:
        --------
        float or array_like
            Form factor value
        """
        if np.isscalar(qm):
            x = qm * qm
            
            # Use series expansion at low q for higher accuracy
            # Double precision: use O(5) Pade with 0.5 cutoff
            if x < 0.5:
                # PadeApproximant[2*Exp[-x^2] + x^2-1)/x^4, {x, 0, 8}]
                A1, A2, A3, A4, A5 = 1./12., 2./99., 1./2640., 1./23760., -1./1995840.
                B1, B2, B3, B4, B5 = 5./12., 5./66., 1./132., 1./2376., 1./95040.
                
                numerator = (((((A5*x + A4)*x + A3)*x + A2)*x + A1)*x + 1.)
                denominator = (((((B5*x + B4)*x + B3)*x + B2)*x + B1)*x + 1.)
                return numerator / denominator
            else:
                # Use full formula for larger x
                return 2.0 * (np.expm1(-x) + x) / (x * x)
        
        else:
            # Vectorized version for arrays
            qm = np.asarray(qm)
            x = qm * qm
            result = np.zeros_like(x)
            
            # Small x regime (use Pade approximant)
            small_mask = x < 0.5
            if np.any(small_mask):
                x_small = x[small_mask]
                A1, A2, A3, A4, A5 = 1./12., 2./99., 1./2640., 1./23760., -1./1995840.
                B1, B2, B3, B4, B5 = 5./12., 5./66., 1./132., 1./2376., 1./95040.
                
                numerator = (((((A5*x_small + A4)*x_small + A3)*x_small + A2)*x_small + A1)*x_small + 1.)
                denominator = (((((B5*x_small + B4)*x_small + B3)*x_small + B2)*x_small + B1)*x_small + 1.)
                result[small_mask] = numerator / denominator
            
            # Large x regime (use full formula)
            large_mask = ~small_mask
            if np.any(large_mask):
                x_large = x[large_mask]
                result[large_mask] = 2.0 * (np.expm1(-x_large) + x_large) / (x_large * x_large)
            
            return result

    def calculate_intensity(self, q, length, kuhn_length, radius, 
                        scale=1.0, background=0.0, pow_scale=0.0, pow_exp=3.0, 
                        gauss_scale=0.10, rm=5):
        """
        Calculate I(q) for flexible cylinder with optional power law and Gaussian coil
        
        Parameters:
        -----------
        q : array_like
            Momentum transfer (1/Å)
        length : float
            Contour length (Å)
        kuhn_length : float
            Kuhn length (Å)
        radius : float
            Cross-sectional radius (Å)
        scale : float
            Scale factor for flexible cylinder. Should be equivalent to I(0)
        background : float
            Constant background
        pow_scale : float
            Power law scale factor
        pow_exp : float
            Power law exponent
        gauss_scale : float
            Scale factor for Gaussian small molecule scattering (dimensionless)
        rm : float
            Radius of gyration for Gaussian small molecule scattering (Å)

        Returns:
        --------
        array_like
            Scattering intensity I(q)
        """
        # Handle scalar/array input
        q = np.asarray(q)
        scalar_input = q.ndim == 0
        q = np.atleast_1d(q)
        
        # Calculate components
        
        cross_section = self.sas_2J1x_x(q * radius)
        #volume = np.pi * radius * radius * length
        gauss = self.gauss_coil(q * rm)

        # Calculate structure factor for each q
        if len(q) == 1:
            flex = self.Sk_WR(q[0], length, kuhn_length)
        else:
            flex = np.array([self.Sk_WR(qi, length, kuhn_length) for qi in q])
        
        # Calculate intensity components
        # 1. Main flexible cylinder term
        flex_intensity = (scale * (cross_section)**2 * flex * 
                        (1 + pow_scale * np.power(q*(1/0.01), -pow_exp)))
        
        # 2. Gaussian coil correction term
        gauss_intensity = (scale/gauss_scale)*(gauss - ((cross_section**2) * flex))

        # 3. Total intensity
        intensity = flex_intensity + gauss_intensity + background

        return float(intensity[0]) if scalar_input else intensity

    def fit_data(self, q_data, intensity_data, uncertainty=None, initial_params=None, bounds=None, fixed_params=None):
        """
        Fit the model to experimental data using least_squares
        
        Parameters:
        -----------
        q_data : array_like
            Experimental q values
        intensity_data : array_like
            Experimental intensities
        uncertainty : array_like, optional
            Uncertainty in intensities (for weighting)
        initial_params : dict, optional
            Initial parameter values
        bounds : dict, optional
            Parameter bounds as {'param': (min, max)}
        fixed_params : dict, optional
            Parameters to keep fixed during fitting
        
        Returns:
        --------
        dict
            Fitted parameters and their uncertainties
        """
        # Default initial parameters
        if initial_params is None:
            initial_params = {
                'length': 1000.0,
                'kuhn_length': 100.0,
                'radius': 20.0,
                'scale': 1.0,
                'background': 0.001,
                'pow_scale': 0.0, 
                'pow_exp': 3.0,
                'gauss_scale': 0.0,
                'rm': 2.0
            }
        
        # Default bounds
        if bounds is None:
            bounds = {
                'length': (10.0, 10000.0),
                'kuhn_length': (10.0, 5000.0),
                'radius': (1.0, 1000.0),
                'scale': (0.001, 1000.0),
                'background': (0.0, 1.0),
                'pow_scale': (0.0, 1.0),
                'pow_exp': (0.0, 4.0),
                'gauss_scale': (0.0, 1000000.0),
                'rm': (0.0, 10.0)
            }
        
        # Handle fixed parameters
        if fixed_params is None:
            fixed_params = {}
        
        # Create list of parameters to fit (not fixed)
        fit_params = {k: v for k, v in initial_params.items() if k not in fixed_params}
        
        # Extract initial values and bounds for fitting parameters only
        param_names = list(fit_params.keys())
        p0 = [fit_params[name] for name in param_names]
        lower_bounds = [bounds[name][0] for name in param_names]
        upper_bounds = [bounds[name][1] for name in param_names]

        # Define residual function for least_squares
        def residuals(params):
            # Combine fitted and fixed parameters
            all_params = fixed_params.copy()
            for i, param_name in enumerate(param_names):
                all_params[param_name] = params[i]
            
            # Calculate model intensity
            model_intensity = self.calculate_intensity(q_data, **all_params)
            
            # Calculate residuals (weighted if uncertainty is provided)
            if uncertainty is not None:
                return (intensity_data - model_intensity) / uncertainty
            else:
                return intensity_data - model_intensity

        # Perform fitting using least_squares
        try:
            result = least_squares(
                residuals, 
                x0=p0,
                bounds=(lower_bounds, upper_bounds),
                max_nfev=5000,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10
            )
            
            # Extract fitted parameters
            popt = result.x
            
            # Calculate parameter uncertainties from Jacobian
            # For least_squares, we need to compute covariance manually
            if result.jac is not None:
                # Compute covariance matrix
                try:
                    # J^T J approximation for covariance
                    JTJ = result.jac.T @ result.jac
                    pcov = np.linalg.inv(JTJ)
                    
                    # Scale by residual variance if no uncertainty provided
                    if uncertainty is None:
                        residual_variance = np.sum(result.fun**2) / (len(q_data) - len(param_names))
                        pcov *= residual_variance
                    
                    param_errors = np.sqrt(np.diag(pcov))
                except np.linalg.LinAlgError:
                    # If covariance calculation fails, use NaN
                    pcov = np.full((len(param_names), len(param_names)), np.nan)
                    param_errors = np.full(len(param_names), np.nan)
            else:
                pcov = np.full((len(param_names), len(param_names)), np.nan)
                param_errors = np.full(len(param_names), np.nan)
            
            # Create result dictionary with all parameters
            fitted_params = fixed_params.copy()
            fitted_param_errors = {name: 0.0 for name in fixed_params.keys()}
            
            for i, name in enumerate(param_names):
                fitted_params[name] = popt[i]
                fitted_param_errors[name] = param_errors[i]
            
            # Calculate goodness of fit
            final_params = fixed_params.copy()
            for i, name in enumerate(param_names):
                final_params[name] = popt[i]
            
            fitted_intensity = self.calculate_intensity(q_data, **final_params)
            residuals_final = intensity_data - fitted_intensity
            
            if uncertainty is not None:
                chi_squared = np.sum((residuals_final / uncertainty)**2)
            else:
                chi_squared = np.sum(residuals_final**2)
            
            results = {
                'fitted_params': fitted_params,
                'param_errors': fitted_param_errors,
                'covariance': pcov,
                'success': result.success,
                'chi_squared': chi_squared,
                'reduced_chi_squared': chi_squared / (len(q_data) - len(param_names)),
                'residuals': residuals_final,
                'fitted_intensity': fitted_intensity,
                'nfev': result.nfev,
                'message': result.message,
                'cost': result.cost
            }
            
            return results
            
        except Exception as e:
            return {
                'fitted_params': initial_params,
                'param_errors': {name: np.nan for name in initial_params.keys()},
                'success': False,
                'error': str(e),
                'covariance': None,
                'chi_squared': np.nan,
                'reduced_chi_squared': np.nan,
                'residuals': None,
                'fitted_intensity': None
            }
    def plot_fit(self, q_data, intensity_data, fit_results, uncertainty=None, q_range=None, title=None):
        """
        Plot experimental data and fitted model
        """
        if not fit_results['success']:
            print(f"Fit failed: {fit_results.get('error', 'Unknown error')}")
            return
        
        # Generate smooth curve for plotting
        if q_range is None:
            q_plot = np.logspace(np.log10(q_data.min()), np.log10(q_data.max()), 200)
        else:
            q_plot = np.logspace(np.log10(q_range[0]), np.log10(q_range[1]), 200)
        
        fitted_params = fit_results['fitted_params']
        intensity_plot = self.calculate_intensity(q_plot, **fitted_params)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Main plot
        if uncertainty is not None:
            ax1.errorbar(q_data, intensity_data, yerr=uncertainty, 
                        fmt='o', markersize=4, capsize=2, label='Data', alpha=0.7, zorder=1)
        else:
            ax1.plot(q_data, intensity_data, 'o', markersize=4, label='Data', alpha=0.7, zorder=1)
        
        ax1.plot(q_plot, intensity_plot, 'r-', linewidth=2, zorder=2,label='Fit')
        ax1.set_xlabel('q (1/Å)')
        ax1.set_ylabel('I(q)')
        ax1.loglog()
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title(title or 'Flexible Cylinder Fit')
        
        # Residuals
        residuals = fit_results['residuals']
        if uncertainty is not None:
            normalized_residuals = residuals / uncertainty
            ax2.errorbar(q_data, normalized_residuals, yerr=1, 
                        fmt='o', markersize=4, capsize=2, alpha=0.7)
            ax2.set_ylabel('(Data - Fit) / σ')
        else:
            ax2.plot(q_data, residuals, 'o', markersize=4, alpha=0.7)
            ax2.set_ylabel('Data - Fit')
        
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('q (1/Å)')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Print fit results
        print("\nFit Results:")
        print("-" * 60)
        for param, value in fitted_params.items():
            error = fit_results['param_errors'][param]
            print(f"{param:15s}: {value:10.6f} ± {error:10.6f}")
        
        if 'reduced_chi_squared' in fit_results:
            print(f"{'Reduced χ²':15s}: {fit_results['reduced_chi_squared']:10.4f}")
        
        # Calculate derived quantities
        L = fitted_params['length']
        b = fitted_params['kuhn_length']
        R = fitted_params['radius']
        
        print(f"\nDerived quantities:")
        print(f"{'L/b ratio':15s}: {L/b:10.2f}")
        print(f"{'Rg (Å)':15s}: {np.sqrt(self.Rgsquare(L, b)):10.1f}")
        print(f"{'Volume (Å³)':15s}: {np.pi * R**2 * L:10.0f}")
        
        plt.show()
        return fig
    
    def test_model(self):
        """Test against reference values and optionally show comparison with SasView"""
        print("Testing Python implementation...")
        
        test_cases = [
            {'q': 0.001, 'params': {'length': 1000.0, 'kuhn_length': 100.0, 'radius': 20.0}, 'expected': 1.052541},
            {'q': 1.0, 'params': {'length': 1000.0, 'kuhn_length': 100.0, 'radius': 20.0}, 'expected': 0.767999},
            {'q': 0.1, 'params': {'length': 10.0, 'kuhn_length': 800.0, 'radius': 2.0}, 'expected': -18.776337},
            {'q': 1.0, 'params': {'length': 100.0, 'kuhn_length': 800.0, 'radius': 50.0}, 'expected': 0.767996}
        ]
        
        print("q\t\tExpected\tCalculated\tRel Error (%)")
        print("-" * 55)
        
        for test in test_cases:
            calculated = self.calculate_intensity(test['q'], **test['params'])
            expected = test['expected']
            rel_error = abs(calculated - expected) / expected * 100

            print(f"{test['q']:8.3f}\t{expected:10.6f}\t{calculated:10.6f}\t{rel_error:8.1f}")

    
    def generate_synthetic_data(self, q_range=(1e-3, 1), n_points=50, noise_level=0.05, **params):
        """
        Generate synthetic SAXS data for testing fitting
        
        Parameters:
        -----------
        q_range : tuple
            (q_min, q_max) range
        n_points : int
            Number of data points
        noise_level : float
            Relative noise level (fraction of intensity)
        **params : 
            Model parameters
        
        Returns:
        --------
        tuple
            (q_data, intensity_data, uncertainty)
        """
        # Default parameters if not provided
        default_params = {
            'length': 800.0,
            'kuhn_length': 120.0,
            'radius': 25.0,
            'sld': 1.5,
            'sld_solvent': 6.3,
            'scale': 1.2,
            'background': 0.01
        }
        default_params.update(params)
        
        # Generate q values
        q_data = np.logspace(np.log10(q_range[0]), np.log10(q_range[1]), n_points)
        
        # Calculate true intensity
        true_intensity = self.calculate_intensity(q_data, **default_params)
        
        # Add realistic noise
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, noise_level * true_intensity)
        intensity_data = true_intensity + noise
        uncertainty = noise_level * true_intensity
        
        return q_data, intensity_data, uncertainty, default_params

# Convenience function for easy import
def create_model():
    """Create a FlexibleCylinderModel instance"""
    return FlexibleCylinderModel()

# Main test function
def main():
    """Test the model"""
    model = FlexibleCylinderModel()
    model.test_model()

if __name__ == "__main__":
    main()