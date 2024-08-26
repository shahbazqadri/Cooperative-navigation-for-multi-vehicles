% plot the planning trajectories
w_max = deg2rad(10);
M = 5;
MPC_horizon = 10;
w_range = linspace(-w_max, w_max, M);
[X, Y, Z ]= meshgrid(w_range, w_range, w_range);
w_set = [X(:) Y(:) Z(:) zeros(M^3, MPC_horizon-3)];%kron(ones(1, MPC_horizon-3), Z(:))];%#zeros(M^3, MPC_horizon-3)];
x0 = [0;0;0];
v = 30;
T = 1;
figure;
for i = 1:M^3
    x_traj = [x0];
    x = x0;
    for j = 1:MPC_horizon
        inputs = [v w_set(i,j)]';
        x = discrete_step(x, inputs, T);
        x_traj = [x_traj x];
    end
    plot(x_traj(1,:), x_traj(2,:),'LineWidth',2);
    hold on
end
grid on;
xlabel('x-meters')
ylabel('y-meters')
axis square

function states = discrete_step(states, inputs, Delta_t)
x = states(1,:);
y = states(2,:);
theta = states(3,:);
v = inputs(1,:);
omega = inputs(2,:);
mu = v*Delta_t*sinc((omega*Delta_t)/(2*pi)); 
x_k1 = x + mu*cos(theta + (omega*Delta_t)/2);
y_k1 = y + mu*sin(theta + (omega*Delta_t)/2);
theta_k1 = theta + omega*Delta_t;

states = [x_k1;y_k1;theta_k1];
end