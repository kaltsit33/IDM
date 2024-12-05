% solution.m
function test(log_kC1, O20, H0)
    % Unit conversion
    transfer = 3.24078e-20; % Conversion factor from km/s/Mpc to 1/Gyr
    kC1 = 10^log_kC1 * transfer;
    O10 = 1 - O20;
    t0 = 1 / H0;
    
    % Solution interval
    tspan = [t0, 0.0];
    
    % Initial value given at t0
    zt0 = [0.0, -H0];
    
    % Solve the ODE
    [t, z] = ode15s(@(t, z) function_(t, z, kC1, O10, H0), tspan, zt0);
    
    % Plot the results
    figure;
    plot(t, z(:, 1), 'DisplayName', 'z(t)');
    xlabel('Time');
    ylabel('Solution');
    legend;
end

function dz = function_(~, z, kC1, O10, H0)
    % z(1) = z(t), z(2) = z'(t)
    dz = zeros(2, 1);
    dz(1) = z(2);
    
    % Reduce the use of parentheses, separate into numerator and denominator
    numerator = ...
        H0^4 * kC1 * O10^2 * (z(1)^4 + 1) + ...
        3 * H0^4 * O10^2 * z(1)^2 * (2 * kC1 - 3 * z(2)) + ...
        H0^4 * O10^2 * z(1)^3 * (4 * kC1 - 3 * z(2)) - ...
        3 * H0^4 * O10^2 * z(2) + ...
        5 * H0^2 * O10 * z(2)^3 - ...
        kC1 * z(2)^4 + ...
        H0^2 * O10 * z(1) * (4 * H0^2 * kC1 * O10 - 9 * H0^2 * O10 * z(2) + 5 * z(2)^3);
    denominator = 2 * H0^2 * O10 * (1 + z(1))^2 * z(2);
    dz(2) = numerator / denominator;
end