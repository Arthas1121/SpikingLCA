
pvp_dict_file_name = argv(){1};
factor_pos = str2num(argv(){2})
factor_neg = str2num(argv(){3})

[d,h] = readpvpfile(pvp_dict_file_name);
d_pos = d
d_neg = d

d_pos{1}.values{1}=max(0,factor_pos*d{1}.values{1})
d_neg{1}.values{1}=max(0,-1*factor_neg*d{1}.values{1});

%d_pos{1}.values{1}=max(0,100*d{1}.values{1})
%d_neg{1}.values{1}=max(0,-1*100*d{1}.values{1});

if (factor_pos != 1)
    writepvpkernelfile([pvp_dict_file_name "_pos_" num2str(factor_pos)],d_pos);
    writepvpkernelfile([pvp_dict_file_name "_neg_" num2str(factor_neg)],d_neg);
else
    writepvpkernelfile([pvp_dict_file_name "_pos"],d_pos);
    writepvpkernelfile([pvp_dict_file_name "_neg"],d_neg);
endif
