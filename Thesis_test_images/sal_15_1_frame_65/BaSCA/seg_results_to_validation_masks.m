% load workspace_after_segmentation.mat

f_idx=2;
num_colonies = length(frame(f_idx).colonyProps);

edge_mask = zeros(frame(f_idx).x,frame(f_idx).y);
inner_mask = zeros(frame(f_idx).x,frame(f_idx).y);
for col_idx =1:num_colonies
    bboxULcorner = frame(f_idx).colonyProps(col_idx).bBoxULCorner;    
    num_cells = size(frame(f_idx).colonyProps(col_idx).cellProps,1);
    for j = 1:num_cells
        
        pixel_perim_bac = frame(f_idx).colonyProps(col_idx).cellProps{j,4}{1};
        for p = 1:length(pixel_perim_bac)
            edge_mask(bboxULcorner(2)+pixel_perim_bac(p,2),bboxULcorner(1)+pixel_perim_bac(p,1))=1;
        end
        
        pixel_bac = frame(f_idx).colonyProps(col_idx).cellProps{j,3}{1};
        for p = 1:length(pixel_bac)
            inner_mask(bboxULcorner(2)+pixel_bac(p,2),bboxULcorner(1)+pixel_bac(p,1))=1;
        end 
        
    end
    
end
inner_mask=inner_mask-edge_mask;

imshow(edge_mask);
figure;
imshow(inner_mask-edge_mask);

imwrite(inner_mask*255,'inner_result.jpg')
imwrite(edge_mask*255,'edge_result.jpg')