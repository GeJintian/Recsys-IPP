ratestring = string(zeros(size(T,1),1));

for i = 1:size(to,1)
   final = to(i);
    
   step = int32(double(string(table2array(T(to(i),4)))));
   first = final - step +int32(1);
   ratestring(i) = rate(T(first:final,:),hotelprop,hotelname,hotel_clicks);
   i;
end

rec = array2table(ratestring,'VariableNames',{'item_recommendations'});