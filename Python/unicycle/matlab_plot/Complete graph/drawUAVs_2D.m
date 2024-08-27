%% prepare the data
close all;
load no_control_ERR_150_1_test_NewNoise.mat
m = 5;
V = 30;
dT = 0.1;

colorcode ='rbgck';
fig = figure(1);
clf;
set (fig, 'Units', 'normalized', 'Position', [0,0,0.3,0.5]);
S = 3500; % plot size
scale = 4;
sim_id = 5;
[Vertices,Faces,facecolors] = defineUAVBody(scale);
ts = 2;
k = ts;
L = length(TRUTH);
alt = -100;
tf = L - 100;
vidfile = VideoWriter('no_control.mp4','MPEG-4');
open(vidfile);
while k < tf
    clf;
    hold on;
    for i = 1:m
        UAV{i}.pn = reshape(TRUTH(sim_id,i,2,k),[],1);
        UAV{i}.pe = reshape(TRUTH(sim_id,i,1,k),[],1);
        UAV{i}.pd = alt;
        UAV{i}.psi = pi/2 - reshape(TRUTH(sim_id,i,3,k),[],1); % heading
        UAV{i}.w = (reshape(TRUTH(sim_id,i,3,k+1),[],1) - reshape(TRUTH(sim_id,i,3,k),[],1))/dT;
        UAV{i}.phi = atan(UAV{i}.w * V/9.8);
        UAV{i}.theta = 0;
        tracker_handle(i) = drawUAV(Vertices,Faces,facecolors,UAV{i},[]);
        plot3(reshape(TRUTH(sim_id,i,1,ts:tf),[],1), reshape(TRUTH(sim_id,i,2,ts:tf),[],1), ones(tf-ts+1,1) * (-alt), [colorcode(i)],'MarkerSize',10)
        plot3(posf(1, i), posf(2,i), -alt, ['d' colorcode(i)],'MarkerSize',10)
    end
    xlabel('East (m)','FontSize',18)
    ylabel('North (m)', 'FontSize',18)
    zlabel('-Down (m)','FontSize',18)
    axis([0,S,0,S,-1,200]);
    % 
    % if k > 1000/V/dT
    %     axis([0,S,0,S,-1,200]);
    % else
    %     axis([0,1000,0,1000,-1,200]);
    % end
    view(2)
    %view(10,45)  % set the view angle for figure
    grid on
    drawnow('update');
    k = k + 10;
    F = getframe(gcf); 
    writeVideo(vidfile,F);
end
close(vidfile);
close all;

%%%%%%%%%%%%%%%%
% define aircraft vertices and faces
function [V,F,colors] = defineUAVBody(scale)


% parameters for drawing aircraft
% scale size
fuse_l1    = 7;
fuse_l2    = 4;
fuse_l3    = 15;
fuse_w     = 2;
wing_l     = 6;
wing_w     = 20;
tail_l     = 3;
tail_h     = 3;
tailwing_w = 10;
tailwing_l = 3;
% colors
red     = [1, 0, 0];
green   = [0, 1, 0];
blue    = [0, 0, 1];
yellow  = [1,1,0];
magenta = [0, 1, 1];


% define vertices and faces for aircraft
V = [...
    fuse_l1,             0,             0;...        % point 1
    fuse_l2,            -fuse_w/2,     -fuse_w/2;... % point 2
    fuse_l2,             fuse_w/2,     -fuse_w/2;... % point 3
    fuse_l2,             fuse_w/2,      fuse_w/2;... % point 4
    fuse_l2,            -fuse_w/2,      fuse_w/2;... % point 5
    -fuse_l3,             0,             0;...        % point 6
    0,                   wing_w/2,      0;...        % point 7
    -wing_l,              wing_w/2,      0;...        % point 8
    -wing_l,             -wing_w/2,      0;...        % point 9
    0,                  -wing_w/2,      0;...        % point 10
    -fuse_l3+tailwing_l,  tailwing_w/2,  0;...        % point 11
    -fuse_l3,             tailwing_w/2,  0;...        % point 12
    -fuse_l3,            -tailwing_w/2,  0;...        % point 13
    -fuse_l3+tailwing_l, -tailwing_w/2,  0;...        % point 14
    -fuse_l3+tailwing_l,  0,             0;...        % point 15
    -fuse_l3+tailwing_l,  0,             -tail_h;...  % point 16
    -fuse_l3,             0,             -tail_h;...  % point 17
    ];

F = [...
    1,  2,  3,  1;... % nose-top
    1,  3,  4,  1;... % nose-left
    1,  4,  5,  1;... % nose-bottom
    1,  5,  2,  1;... % nose-right
    2,  3,  6,  2;... % fuselage-top
    3,  6,  4,  3;... % fuselage-left
    4,  6,  5,  4;... % fuselage-bottom
    2,  5,  6,  2;... % fuselage-right
    7,  8,  9, 10;... % wing
    11, 12, 13, 14;... % tailwing
    6, 15, 17, 17;... % tail

    ];

colors = [...
    yellow;... % nose-top
    yellow;... % nose-left
    yellow;... % nose-bottom
    yellow;... % nose-right
    blue;... % fuselage-top
    blue;... % fuselage-left
    red;... % fuselage-bottom
    blue;... % fuselage-right
    green;... % wing
    green;... % tailwing
    blue;... % tail
    ];

V = scale*V;   % rescale vertices

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function new_handle = drawUAV(V,F,colors,UAV,handle)
V = rotate(V', UAV.phi, UAV.theta, UAV.psi)';  % rotate rigid body
V = translate(V', UAV.pn, UAV.pe, -100)';  % translate after rotation

% transform vertices from NED to XYZ (for matlab rendering)
R = [...
    0, 1, 0;...
    1, 0, 0;...
    0, 0, -1;...
    ];
V = V*R;

if isempty(handle),
    new_handle = patch('Vertices', V, 'Faces', F,...
        'FaceVertexCData',colors,...
        'FaceColor','flat');
else
    set(handle,'Vertices',V,'Faces',F);
end

end


%%%%%%%%%%%%%%%%%%%%%%%
function XYZ=rotate(XYZ,phi,theta,psi)
% define rotation matrix
R_roll = [...
    1, 0, 0;...
    0, cos(phi), -sin(phi);...
    0, sin(phi), cos(phi)];
R_pitch = [...
    cos(theta), 0, sin(theta);...
    0, 1, 0;...
    -sin(theta), 0, cos(theta)];
R_yaw = [...
    cos(psi), -sin(psi), 0;...
    sin(psi), cos(psi), 0;...
    0, 0, 1];

% rotate vertices
XYZ = R_yaw*R_pitch*R_roll*XYZ;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% translate vertices by pn, pe, pd
function XYZ = translate(XYZ,pn,pe,pd)

XYZ = XYZ + repmat([pn;pe;pd],1,size(XYZ,2));

end

