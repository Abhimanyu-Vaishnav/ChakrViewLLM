# Gradient Descent Convergence and Momentum Mechanics

## 1. Optimization Dynamics
- Nesterov accelerated gradient: Look-ahead momentum updates prevent overshooting along high-curvature ravines.
- Adam optimizer formulation: Exponential moving averages of first and second gradient moments guide parameter updates.
- Learning rate warmup: Linear step increases stabilize early optimization trajectory before cosine learning rate decay.
- Weight decay decoupling: True AdamW separates L2 regularization penalties from adaptive gradient moving averages.
- Polyak averaging: Exponential moving average of model checkpoints yields smoother optimization loss landscapes.
