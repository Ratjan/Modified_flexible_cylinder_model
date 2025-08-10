# Modified_flexible_cylinder_model

Early attempt at Python implementation of the modified flexible cylinder model developed by J.S. Pedersen presented in: 
Correlation between arabinose content and the conformation of arabinoxylan in water dispersions
https://doi.org/10.1016/j.carbpol.2025.124082.

Source code is based on the C implementation of the models from the sasview documentation. 
The model uses modifed versions of the sasview flexible cylinder and mono gauss coil models.
As the models are highly complicated, Claude Sonnet 4 was used to help in the porting process.

Fitting optimization is performed using scipy least_squares, therefore the fitted results will differ to some extent compared to those fitted using the WLSQSAXS software by Oliveira JCP and Pedersen JS.

Next goal will be to add the PRISM rod structure factor.
