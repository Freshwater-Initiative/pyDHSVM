% Convert Chehalis River Basin DHSVM maps into grids
%Christina Bandaragoda 05/04/16

% Sample output conversion
%Livneh 2003-2013  
%./myconvert float ascii snowoutput_8_2003-2013Map.Snow.Tsurf.bin Snow.Tsurf409.m.asc 40392 736
%NNRP 2003-2010 
%./myconvert float ascii snowoutput_8_2003-2013Map.Snow.Tsurf.bin Snow.Tsurf409.m.asc 29376 736

%Header
% ncols	 736
% nrows	 918
% xllcorner	434265.00000
% yllcorner	5129745.00000
% cellsize	150.00000
% NODATA_value	-9999

nrow=918;

%Print Livneh data
cd C:\Users\cband\Chehalis\ModelOutputs\snowoutputLivneh_8_2003-2013

i=2003
datafilename404={strcat('SnowSwq404m_0101',num2str(i)),...
  strcat('SnowSwq404m_0201',num2str(i)),...
     strcat('SnowSwq404m_0301',num2str(i)),...
  strcat('SnowSwq404m_0401',num2str(i))};
for i=2004:2013
datafilename404=[datafilename404 {strcat('SnowSwq404m_0101',num2str(i)),...
  strcat('SnowSwq404m_0201',num2str(i)),...
     strcat('SnowSwq404m_0301',num2str(i)),...
  strcat('SnowSwq404m_0401',num2str(i))}];
end

i=2003
datafilename409={strcat('SnowSnowTsurf409C_0101',num2str(i)),...
  strcat('SnowSnowTsurf409C_0201',num2str(i)),...
     strcat('SnowSnowTsurf409C_0301',num2str(i)),...
  strcat('SnowSnowTsurf409C_0401',num2str(i))};
for i=2004:2013
datafilename409=[datafilename409 {strcat('SnowSnowTsurf409C_0101',num2str(i)),...
  strcat('SnowSnowTsurf409C_0201',num2str(i)),...
     strcat('SnowSnowTsurf409C_0301',num2str(i)),...
  strcat('SnowSnowTsurf409C_0401',num2str(i))}];
end



GCM='Livneh';
snowdata=load('Snow.Swq404.m.asc');
snowTdata=load('Snow.Tsurf409.m.asc');

%Print snow data
%for i=1:2 
dlmwrite(strcat(char(GCM),'_',char(datafilename404(1)),'.asc'),snowdata(1:nrow,:),'delimiter',' ','precision','%.1f');
dlmwrite(strcat(char(GCM),'_',char(datafilename404(2)),'.asc'),snowdata(nrow+1:nrow*2,:),'delimiter',' ','precision','%.1f');

for i=3:length(datafilename404)
dlmwrite(strcat(char(GCM),'_',char(datafilename404(i)),'.asc'),snowdata(nrow*(i-1)+1:nrow*i,:),'delimiter',' ','precision','%.1f');
end

%Print snow T data
dlmwrite(strcat(char(GCM),'_',char(datafilename409(1)),'.asc'),snowTdata(1:nrow,:),'delimiter',' ','precision','%.1f');
dlmwrite(strcat(char(GCM),'_',char(datafilename409(2)),'.asc'),snowTdata(nrow+1:nrow*2,:),'delimiter',' ','precision','%.1f');

for i=3:length(datafilename409)
dlmwrite(strcat(char(GCM),'_',char(datafilename409(i)),'.asc'),snowTdata(nrow*(i-1)+1:nrow*i,:),'delimiter',' ','precision','%.1f');
end

%Print WRF NNRP data
cd C:\Users\cband\Chehalis\ModelOutputs\snowoutputNNRP_8_2003-2013


i=2003
datafilename404={strcat('SnowSwq404m_0101',num2str(i)),...
  strcat('SnowSwq404m_0201',num2str(i)),...
     strcat('SnowSwq404m_0301',num2str(i)),...
  strcat('SnowSwq404m_0401',num2str(i))};
for i=2004:2010
datafilename404=[datafilename404 {strcat('SnowSwq404m_0101',num2str(i)),...
  strcat('SnowSwq404m_0201',num2str(i)),...
     strcat('SnowSwq404m_0301',num2str(i)),...
  strcat('SnowSwq404m_0401',num2str(i))}];
end

i=2003
datafilename409={strcat('SnowSnowTsurf409C_0101',num2str(i)),...
  strcat('SnowSnowTsurf409C_0201',num2str(i)),...
     strcat('SnowSnowTsurf409C_0301',num2str(i)),...
  strcat('SnowSnowTsurf409C_0401',num2str(i))};
for i=2004:2010
datafilename409=[datafilename409 {strcat('SnowSnowTsurf409C_0101',num2str(i)),...
  strcat('SnowSnowTsurf409C_0201',num2str(i)),...
     strcat('SnowSnowTsurf409C_0301',num2str(i)),...
  strcat('SnowSnowTsurf409C_0401',num2str(i))}];
end

GCM='WRF_NNRP';
snowdata=load('Snow.Swq404.m.asc');
snowTdata=load('Snow.Tsurf409.C.asc');

%Print snow data
%for i=1:2 
dlmwrite(strcat(char(GCM),'_',char(datafilename404(1)),'.asc'),snowdata(1:nrow,:),'delimiter',' ','precision','%.1f');
dlmwrite(strcat(char(GCM),'_',char(datafilename404(2)),'.asc'),snowdata(nrow+1:nrow*2,:),'delimiter',' ','precision','%.1f');

for i=3:length(datafilename404)
dlmwrite(strcat(char(GCM),'_',char(datafilename404(i)),'.asc'),snowdata(nrow*(i-1)+1:nrow*i,:),'delimiter',' ','precision','%.1f');
end

%Print snow T data
dlmwrite(strcat(char(GCM),'_',char(datafilename409(1)),'.asc'),snowTdata(1:nrow,:),'delimiter',' ','precision','%.1f');
dlmwrite(strcat(char(GCM),'_',char(datafilename409(2)),'.asc'),snowTdata(nrow+1:nrow*2,:),'delimiter',' ','precision','%.1f');

for i=3:length(datafilename409)
dlmwrite(strcat(char(GCM),'_',char(datafilename409(i)),'.asc'),snowTdata(nrow*(i-1)+1:nrow*i,:),'delimiter',' ','precision','%.1f');
end

SaveAsciiRaster(varname, header)
