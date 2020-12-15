function out=Optimize_clement(parameters,...
    model,ytrue)


X_test=(parameters);

zz = (predict(model,X_test'))';
%zz=clfy.inverse_transform(zz);
zz=reshape(zz,[],6);

%%

ytrue=reshape(ytrue,[],6);
Hardmean=double(reshape(zz,[],1));
gg=size(ytrue,1);
% a1=((ytrue'-Hardmean').^2).^0.5;
a1=(1/(2*gg)) * sum((ytrue-Hardmean).^2);
%a1=sum((ytrue-Hardmean).^2);

%a1=abs((ytrue-Hardmean));

cc = sum(a1);
out=cc;

end