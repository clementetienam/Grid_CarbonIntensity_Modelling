function clement2 = DCTsigned(sg, N)
%% DCT domain reduction for permeability field
disp('  DCT domain reduction for permeability  ');
%%PhD student: Clement Oku Etienam
%%Supervisor: Dr Rossmary Villegas
%%Co-supervisor: Dr Masoud Babei
%%Collaborator : Dr Oliver Dorn
disp('  Load the relevant files  ');

sgsim=(sg);
LF=reshape(sgsim,36000,N);
disp('  carry out DCT domain reduction  ');
for ii=1:N
    lf=reshape(LF(:,ii),120,60,5);
     for jj=1:5
         value=lf(:,:,jj);
         usdf=mirt_dctn(value);
         usdf=reshape(usdf,7200,1);
         young(:,jj)=usdf;
     end
      sdfbig=reshape(young,36000,1);
  clement(:,ii)=sdfbig;
end
disp('  extract the significant DCT coefficients  ');
clement=reshape(clement,36000,N);
for iii=1:N
    lf2=reshape(clement(:,iii),120,60,5);
for jjj=1:5
    val1=lf2(1:60,1:30,jjj);
    val1=reshape(val1,1800,1);
   val2(:,jjj)=val1;
end
  sdfbig2=reshape(val2,9000,1);
  clement2(:,iii)=sdfbig2;
end


end