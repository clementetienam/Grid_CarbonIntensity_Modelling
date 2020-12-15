function  ouut=Tackit(forecastFromSelf, Truee)
mean_uchee=sum(forecastFromSelf);
divideit=(mean_uchee./(Truee));

ouut=[];
parfor i=1:size(forecastFromSelf,1)
ouut(i,:)=forecastFromSelf(i,:)./divideit;
end

end