%% main
clear;clc;
close all;
photopath = 'photo\';
[filename, pathname, index] = uigetfile([photopath,'*.bmp;*.jpg;*.png']);
test_image = im2double(imread([pathname,filename]));
n = 128;m = 128;% size of the image
%% locate the four corner of the image
imshow([test_image(:,:,:)]);hold on;
for p = 1:4
    [loc_x(p),loc_y(p)] = ginput(1);
    plot(loc_x(p),loc_y(p),'r.');
end
loc_x = floor(loc_x);
loc_y = floor(loc_y);
%% sort the corner
[X,Y] = my_sort(loc_x,loc_y,test_image);
%% perspective transformation
img = imread(strcat(pathname,filename));
I = my_pres_trans(img,X,Y,m,n);
%% write the image
k = find(filename == '.');
pathfile=[pathname,[filename(1:k-1),'_re.png']];
imwrite(I,pathfile);

 