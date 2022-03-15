function [reference] = rate( subtable,hotelprop,hotelname,hotel_clicks)
%input a subtable from the step 1 to the final step and out put one single
%recommendation result.

%My hotel prop is a n*155 matrix,(155 features included,logical)
%hotel name is the corresponding n*1 matrix int32

    
    
        imp = int32(double(split( string(table2array(subtable(end,11))),"|")));%get the impression from the final line
        ref = int32(double(string(table2array(subtable(:,6)))));%get the previous reference
        
        len = size(imp,1);%get the length of the submatrix 
    batch_hotel_filter = zeros(len,size(hotelprop,2));% create a matrix used to store 1 * 155 hotel vector 
        
       [Lia , Locb] = ismember(imp,hotelname); %Use locb to locate the location of the impressions in hotelname
                                               %which is also the location
                                               %in hotelprop
       
       for i = 1:size(Lia,1)
           if Lia(i)
                batch_hotel_filter(i,:) = (hotelprop(Locb(i),:)+.1);%fill in the 1*155 vectors 
           end                                                      %from hotelprop with the locb
       end
       
        [Lia , Locb] = ismember(unique(ref),hotelname);
       if sum(Lia)~=0
        Locb = Locb(Lia);
        Lia = Lia(Lia);
        batch_ref_filter = zeros(1,size(hotelprop,2));%get the reference vectors from the impression vector
       
        %calculate the vector used in knn
       for i = 1:size(Lia,1)
           if Lia(i)
                batch_ref_filter(1,:) = batch_ref_filter(1,:) + (2^i).* (hotelprop(Locb(i),:)); 
           end
       end      
       
      k = knnsearch(batch_hotel_filter,batch_ref_filter,'K',size(imp,1),'Distance','cosine');
      reference = join(string(imp(k)')," ");
       end
       
       if sum(Lia==0)
           pris = int32(double(split( string(table2array(subtable(end,12))),"|")));
           coldstart = zeros(len,4);
           [Lia , Locb] = ismember(imp,hotel_clicks(:,1));
        for i = 1:size(Lia,1)
           if Lia(i)
                coldstart(i,1:2) = hotel_clicks(Locb(i),:);%fill in the 1*155 vectors 
           end                                                      %from hotelprop with the locb
        end
       coldstart(:,3) = pris;
       theta = [0.1898 ,-0.0727];
       coldstart(:,2:3) = coldstart(:,2:3)./[22.7993,163.8196];
       coldstart(:,4) = coldstart(:,2:3)*theta';
       [~,idx] = sort(coldstart(:,4),'descend'); % sort just the first column
        sortedmat = coldstart(idx,:);   % sort the whole matrix using the sort indices
        reference = join(string(sortedmat(:,1)')," ");
       end
      %reference is the answer.
end

