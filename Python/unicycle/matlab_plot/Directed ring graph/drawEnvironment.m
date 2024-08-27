function drawEnvironment(uu,P)

    % process inputs uu
    NN = 0;
    tracker.pn           = uu(1+NN);       % inertial North position     
    tracker.pe           = uu(2+NN);       % inertial East position
    tracker.pd           = uu(3+NN);       % inertial Down position
    %tracker.u           = uu(4+NN);       % body frame velocities
    %tracker.v           = uu(5+NN);       
    %tracker.w           = uu(6+NN);       
    tracker.phi          = uu(7+NN);       % roll angle         
    tracker.theta        = uu(8+NN);       % pitch angle     
    tracker.psi          = uu(9+NN);       % yaw angle     
    %tracker.p           = uu(10+NN);      % roll rate
    %tracker.q           = uu(11+NN);      % pitch rate     
    %tracker.r           = uu(12+NN);      % yaw rate    
    tracker.az           = uu(13+NN);      % gimbal azimuth angle
    tracker.el           = uu(14+NN);      % gimbal elevation angle
    NN = NN + 14;
    tracker.camera.eps_x = uu(1+NN);       % x pixel location of target in tracker's camera
    tracker.camera.eps_y = uu(2+NN);       % y pixel location of target in tracker's camera
    tracker.camera.eps_s = uu(3+NN);       % pixel size of target in tracker's camera
    tracker.camera.az    = uu(4+NN);
    tracker.camera.el    = uu(5+NN);
    NN = NN + 5;
    tracker.path.flag    = uu(1+NN);      % path flag
    tracker.path.r       = [uu(3+NN); uu(4+NN); uu(5+NN)];
    tracker.path.q       = [uu(6+NN); uu(7+NN); uu(8+NN)];
    tracker.path.c       = [uu(9+NN); uu(10+NN); uu(11+NN)];
    tracker.path.rho     = uu(12+NN);
    tracker.path.lam     = uu(13+NN);
    NN = NN + 13;
    handoff.pn           = uu(1+NN);       % inertial North position     
    handoff.pe           = uu(2+NN);       % inertial East position
    handoff.pd           = uu(3+NN);       % inertial Down position
    %handoff.u           = uu(4+NN);       % body frame velocities
    %handoff.v           = uu(5+NN);       
    %handoff.w           = uu(6+NN);       
    handoff.phi          = uu(7+NN);       % roll angle         
    handoff.theta        = uu(8+NN);       % pitch angle     
    handoff.psi          = uu(9+NN);       % yaw angle     
    %handoff.p           = uu(10+NN);      % roll rate
    %handoff.q           = uu(11+NN);      % pitch rate     
    %handoff.r           = uu(12+NN);      % yaw rate    
    handoff.az           = uu(13+NN);      % gimbal azimuth angle
    handoff.el           = uu(14+NN);      % gimbal elevation angle
    NN = NN + 14;
    handoff.camera.eps_x = uu(1+NN);       % x pixel location of target in tracker's camera
    handoff.camera.eps_y = uu(2+NN);       % y pixel location of target in tracker's camera
    handoff.camera.eps_s = uu(3+NN);       % pixel size of target in tracker's camera
    handoff.camera.az    = uu(4+NN);
    handoff.camera.el    = uu(5+NN);
    NN = NN + 5;
    handoff.forward_camera.eps_x = uu(1+NN);  % x pixel location of tracker in handoff fixed camera
    handoff.forward_camera.eps_y = uu(2+NN);  % y pixel location of tracker in handoff fixed camera
    handoff.forward_camera.eps_s = uu(3+NN);  % pixel size of tracker in handoff fixed camera
    NN = NN + 3;
    handoff.path.flag    = uu(1+NN);      % path flag
    handoff.path.r       = [uu(3+NN); uu(4+NN); uu(5+NN)];
    handoff.path.q       = [uu(6+NN); uu(7+NN); uu(8+NN)];
    handoff.path.c       = [uu(9+NN); uu(10+NN); uu(11+NN)];
    handoff.path.rho     = uu(12+NN);
    handoff.path.lam     = uu(13+NN);
    NN = NN + 13;
    target.pn            = uu(1+NN);       % inertial North position     
    target.pe            = uu(2+NN);       % inertial East position
    target.pd            = uu(3+NN);       % inertial Down position
    target.u             = uu(4+NN);       % body frame velocities
    target.v             = uu(5+NN);       
    target.w             = uu(6+NN);       
    NN = NN + 6;
    t                    = uu(1+NN);      % time
    
    % define persistent variables 
    persistent tracker_handle;                  % figure handle for tracker
    persistent tracker_fov_handle;              % handle for tracker FOV
    persistent tracker_camera_handle;           % handle for target in tracker camera
    persistent tracker_path_handle;             % handle for straight-line or orbit path
    persistent handoff_handle;                  % handle for handoff UAS
    persistent handoff_fov_handle;              % handle for tracker FOV
    persistent handoff_camera_handle;           % handle for target in handoff camera
    persistent handoff_forward_camera_handle;   % handle for target in handoff camera
    persistent handoff_path_handle;             % handle for straight-line or orbit path
    persistent target_handle;                   % handle for target
    persistent Faces
    persistent Vertices
    persistent facecolors

    S = 500; % plot size
    
    % first time function is called, initialize plot and persistent vars
    if t==0,

        fig=figure(1);
        clf;
        set (fig, 'Units', 'normalized', 'Position', [0,0,1,1]);
        subplot(2,4,[1:2 5:6])
            scale = 4;
            [Vertices,Faces,facecolors] = defineUAVBody(scale);                              
            tracker_handle = drawUAV(Vertices,Faces,facecolors,tracker,[]);
            hold on
            tracker_fov_handle = drawFov(tracker,P.cam_fov,[]);
            tracker_path_handle = drawPath(tracker.path, S/2, [], 'r');
            handoff_handle = drawUAV(Vertices,Faces,facecolors,handoff,[]);
            handoff_fov_handle = drawFov(handoff,P.cam_fov,[]);
            handoff_path_handle = drawPath(handoff.path, S/2, [], 'g');
            target_handle = drawTarget(target, S/50, []);
  
            title('Target Handoff Scenario')
            xlabel('East')
            ylabel('North')
            zlabel('-Down')
            axis([-S,S,-S,S,-1,S/2]);
            view(-17,24)  % set the view angle for figure
            grid on
 
        subplot(2,4,3)
            tracker_camera_handle = drawBlob([tracker.camera.eps_y, tracker.camera.eps_x], tracker.camera.eps_s, []);
            axis([-P.cam_pix/2,P.cam_pix/2,-P.cam_pix/2,P.cam_pix/2])
            title('Tracking UAS Camera View')
            xlabel('px (pixels)')
            ylabel('py (pixels)')
 
        subplot(2,4,7)
            handoff_camera_handle = drawBlob([handoff.camera.eps_y, handoff.camera.eps_x], handoff.camera.eps_s, []);
            axis([-P.cam_pix/2,P.cam_pix/2,-P.cam_pix/2,P.cam_pix/2])
            title('Handoff UAS Camera View')
            xlabel('px (pixels)')
            ylabel('py (pixels)')
 
        subplot(2,4,8)
            handoff_forward_camera_handle = drawBlob([handoff.forward_camera.eps_y,...
                handoff.forward_camera.eps_x], handoff.forward_camera.eps_s, []);
            axis([-P.cam_pix/2,P.cam_pix/2,-P.cam_pix/2,P.cam_pix/2])
            title('Handoff Forward Camera View')
            xlabel('px (pixels)')
            ylabel('py (pixels)')

        %drawnow limitrate
        drawnow('update');
 
    % at every other time step, redraw all vehicles
    else
        drawUAV(Vertices,Faces,facecolors,tracker,tracker_handle);
        drawFov(tracker,P.cam_fov,tracker_fov_handle);
        drawPath(tracker.path, S/2, tracker_path_handle);
        drawUAV(Vertices,Faces,facecolors,handoff,handoff_handle);
        drawFov(handoff,P.cam_fov,handoff_fov_handle);
        drawPath(handoff.path, S/2, handoff_path_handle);
        drawTarget(target, S/50, target_handle);
        drawBlob([tracker.camera.eps_y, tracker.camera.eps_x], tracker.camera.eps_s, tracker_camera_handle);
        drawBlob([handoff.camera.eps_y, handoff.camera.eps_x], handoff.camera.eps_s, handoff_camera_handle);
        drawBlob([handoff.forward_camera.eps_y, handoff.forward_camera.eps_x],...
            handoff.forward_camera.eps_s, handoff_forward_camera_handle);
 
        %drawnow limitrate
        drawnow('update');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function new_handle = drawPath(path, S, handle, color)
 
    switch path.flag,
        case 1,
            XX = [path.r(1), path.r(1)+S*path.q(1)];
            YY = [path.r(2), path.r(2)+S*path.q(2)];
            ZZ = [path.r(3), path.r(3)+S*path.q(3)];
        case 2,
            N = 100;
            th = [0:2*pi/N:2*pi];
            XX = path.c(1) + path.rho*cos(th);
            YY = path.c(2) + path.rho*sin(th);
            ZZ = path.c(3)*ones(size(th));
    end
    
    if isempty(handle),
        new_handle = plot3(YY,XX,-ZZ,color);
    else
        set(handle,'XData', YY, 'YData', XX, 'ZData', -ZZ);
    end
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function new_handle=drawBlob(z, R, handle)

  th = 0:.1:2*pi;
  X = z(1)+ R*cos(th);
  Y = z(2)+ R*sin(th);
  
  if isempty(handle),
    new_handle = fill(Y, X, 'r');
  else
    set(handle,'XData',Y,'YData',X);
  end
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function new_handle = drawFov(UAV,cam_fov,handle)
                           

    %-------vertices and faces for camera field-of-view --------------
    % vertices
    % define unit vectors along fov in the camera gimbal frame
    pts = [...
        cos(cam_fov/2)*cos(cam_fov/2),  sin(cam_fov/2)*cos(cam_fov/2), -sin(cam_fov/2);...
        cos(cam_fov/2)*cos(cam_fov/2), -sin(cam_fov/2)*cos(cam_fov/2), -sin(cam_fov/2);...
        cos(cam_fov/2)*cos(cam_fov/2), -sin(cam_fov/2)*cos(cam_fov/2),  sin(cam_fov/2);...
        cos(cam_fov/2)*cos(cam_fov/2),  sin(cam_fov/2)*cos(cam_fov/2),  sin(cam_fov/2);...
        ]';
    % transform from gimbal coordinates to the vehicle coordinates
    pts = Rot_v_to_b(UAV.phi,UAV.theta,UAV.psi)'*Rot_b_to_g(UAV.az,UAV.el)'*pts;

    % first vertex is at center of MAV vehicle frame
    Vert = [UAV.pn, UAV.pe, UAV.pd];  
    % project field of view lines onto ground plane and make correction
    % when the projection is above the horizon
    for i=1:4,
        % alpha is the angle that the field-of-view line makes with horizon
        alpha = atan2(pts(3,i),norm(pts(1:2,i)));
        if alpha > 0,
            % this is the normal case when the field-of-view line
            % intersects ground plane
            Vert = [...
                Vert;...
                [UAV.pn-UAV.pd*pts(1,i)/pts(3,i), UAV.pe-UAV.pd*pts(2,i)/pts(3,i), 0];...
                ];
        else
            % this is when the field-of-view line is above the horizon.  In
            % this case, extend to a finite, but far away (9999) location.
            Vert = [...
                Vert;...
                [UAV.pn+9999*pts(1,i), UAV.pe+9999*pts(2,i),0];...
            ];
        end
    end

    Faces = [...
          1, 1, 2, 2;... % x-y face
          1, 1, 3, 3;... % x-y face
          1, 1, 4, 4;... % x-y face
          1, 1, 5, 5;... % x-y face
          2, 3, 4, 5;... % x-y face
        ];

    edgecolor      = [0,0,0]; % black
    footprintcolor = [1,1,1];%[1,0,1];%[1,1,0];
    colors = [edgecolor; edgecolor; edgecolor; edgecolor; footprintcolor];  

  % transform vertices from NED to XYZ (for matlab rendering)
  R = [...
      0, 1, 0;...
      1, 0, 0;...
      0, 0, -1;...
      ];
  Vert = Vert*R;

  if isempty(handle),
    new_handle = patch('Vertices', Vert, 'Faces', Faces,...
                 'FaceVertexCData',colors,...
                 'FaceColor','flat');
  else
    set(handle,'Vertices',Vert,'Faces',Faces);
  end
  
end 

%%%%%%%%%%%%%%%%%%%%%%%
function R = Rot_v_to_b(phi,theta,psi);
% Rotation matrix from body coordinates to vehicle coordinates

Rot_v_to_v1 = [...
    cos(psi), sin(psi), 0;...
    -sin(psi), cos(psi), 0;...
    0, 0, 1;...
    ];
    
Rot_v1_to_v2 = [...
    cos(theta), 0, -sin(theta);...
    0, 1, 0;...
    sin(theta), 0, cos(theta);...
    ];
    
Rot_v2_to_b = [...
    1, 0, 0;...
    0, cos(phi), sin(phi);...
    0, -sin(phi), cos(phi);...
    ];
    
R = Rot_v2_to_b * Rot_v1_to_v2 * Rot_v_to_v1;

end

%%%%%%%%%%%%%%%%%%%%%%%
function R = Rot_b_to_g(az,el);
% Rotation matrix from body coordinates to gimbal coordinates
Rot_b_to_g1 = [...
    cos(az), sin(az), 0;...
    -sin(az), cos(az), 0;...
    0, 0, 1;...
    ];

Rot_g1_to_g = [...
    cos(el), 0, -sin(el);...
    0, 1, 0;...
    sin(el), 0, cos(el);...
    ];

R = Rot_g1_to_g * Rot_b_to_g1;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function new_handle=drawTarget(target, R, handle)
  th = 0:.1:2*pi;
  X = target.pn + R*cos(th);
  Y = target.pe + R*sin(th);
  Z = target.pd*ones(length(th));
  
  if isempty(handle),
    new_handle = fill(Y, X, 'r');
  else
    set(handle,'XData',Y,'YData',X);
  end
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function new_handle = drawUAV(V,F,colors,UAV,handle)
  V = rotate(V', UAV.phi, UAV.theta, UAV.psi)';  % rotate rigid body  
  V = translate(V', UAV.pn, UAV.pe, UAV.pd)';  % translate after rotation

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function handle = drawPath(path, S, handle, mode)
%     flag = path(1); 
%     r    = [path(3); path(4); path(5)];
%     q    = [path(6); path(7); path(8)];
%     c    = [path(9); path(10); path(11)];
%     rho  = path(12);
%     lam  = path(13);
% 
%     switch flag,
%         case 1,
%             XX = [r(1), r(1)+S*q(1)];
%             YY = [r(2), r(2)+S*q(2)];
%             ZZ = [r(3), r(3)+S*q(3)];
%         case 2,
%             N = 100;
%             th = [0:2*pi/N:2*pi];
%             XX = c(1) + rho*cos(th);
%             YY = c(2) + rho*sin(th);
%             ZZ = c(3)*ones(size(th));
%     end
%     
%     if isempty(handle),
%         handle = plot3(YY,XX,-ZZ,'r', 'EraseMode', mode);
%     else
%         set(handle,'XData', YY, 'YData', XX, 'ZData', -ZZ);
%         drawnow
%     end
% end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function handle = drawWaypoints(waypoints, R_min, handle, mode)

    if waypoints(1,4)==-9999, % check to see if Dubins paths
        XX = [waypoints(:,1)];
        YY = [waypoints(:,2)];
        ZZ = [waypoints(:,3)];
    else
        XX = [];
        YY = [];
        for i=2:size(waypoints,1),
            dubinspath = dubinsParameters(waypoints(i-1,:),waypoints(i,:),R_min);
            [tmpX,tmpY] = pointsAlongDubinsPath(dubinspath,0.1);
            XX = [XX; tmpX];
            YY = [YY; tmpY];     
        end
        ZZ = waypoints(i,3)*ones(size(XX));
    end
    
    if isempty(handle),
        handle = plot3(YY,XX,-ZZ,'b', 'EraseMode', mode);
    else
        set(handle,'XData', YY, 'YData', XX, 'ZData', -ZZ);
        drawnow
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% pointsAlongDubinsPath
%%   Find points along Dubin's path separted by Del (to be used in
%%   collision detection)
function [X,Y] = pointsAlongDubinsPath(dubinspath,Del)


  % points along start circle
  th1 = mod(atan2(dubinspath.ps(2)-dubinspath.cs(2),dubinspath.ps(1)-dubinspath.cs(1)),2*pi);
  th2 = mod(atan2(dubinspath.w1(2)-dubinspath.cs(2),dubinspath.w1(1)-dubinspath.cs(1)),2*pi);
  if dubinspath.lams>0,
      if th1>=th2,
        th = [th1:Del:2*pi,0:Del:th2];
      else
        th = [th1:Del:th2];
      end
  else
      if th1<=th2,
        th = [th1:-Del:0,2*pi:-Del:th2];
      else
        th = [th1:-Del:th2];
      end
  end
  X = [];
  Y = [];
  for i=1:length(th),
    X = [X; dubinspath.cs(1)+dubinspath.R*cos(th(i))]; 
    Y = [Y; dubinspath.cs(2)+dubinspath.R*sin(th(i))];
  end
  
  % points along straight line 
  sig = 0;
  while sig<=1,
      X = [X; (1-sig)*dubinspath.w1(1) + sig*dubinspath.w2(1)];
      Y = [Y; (1-sig)*dubinspath.w1(2) + sig*dubinspath.w2(2)];
      sig = sig + Del;
  end
    
  % points along end circle
  th2 = mod(atan2(dubinspath.pe(2)-dubinspath.ce(2),dubinspath.pe(1)-dubinspath.ce(1)),2*pi);
  th1 = mod(atan2(dubinspath.w2(2)-dubinspath.ce(2),dubinspath.w2(1)-dubinspath.ce(1)),2*pi);
  if dubinspath.lame>0,
      if th1>=th2,
        th = [th1:Del:2*pi,0:Del:th2];
      else
        th = [th1:Del:th2];
      end
  else
      if th1<=th2,
        th = [th1:-Del:0,2*pi:-Del:th2];
      else
        th = [th1:-Del:th2];
      end
  end
  for i=1:length(th),
    X = [X; dubinspath.ce(1)+dubinspath.R*cos(th(i))]; 
    Y = [Y; dubinspath.ce(2)+dubinspath.R*sin(th(i))];
  end
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
  