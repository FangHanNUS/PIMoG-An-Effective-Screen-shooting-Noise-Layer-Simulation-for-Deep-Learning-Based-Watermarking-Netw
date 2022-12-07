function [X Y] = my_sort(loc_x,loc_y,test_image)
n = max(loc_x)+min(loc_x);
m = max(loc_y)+min(loc_y);
for i = 1:4
    if loc_x(i) < n/2 & loc_y(i) < m/2
        Y(1) = loc_y(i);
        X(1) = loc_x(i);
    elseif loc_x(i) < n/2 & loc_y(i) > m/2
        Y(4) = loc_y(i);
        X(4) = loc_x(i);
    elseif loc_x(i) > n/2 & loc_y(i) > m/2
        Y(3) = loc_y(i);
        X(3) = loc_x(i);
    elseif loc_x(i) > n/2 & loc_y(i) < m/2
        Y(2) = loc_y(i);
        X(2) = loc_x(i);
    end
end
X = X';
Y = Y';
