set style data dots
set nokey
set xrange [0:13.04681]
set yrange [ 11.06690 : 28.10945]
set arrow from  1.90371,  11.06690 to  1.90371,  28.10945 nohead
set arrow from  3.55237,  11.06690 to  3.55237,  28.10945 nohead
set arrow from  4.50422,  11.06690 to  4.50422,  28.10945 nohead
set arrow from  5.85034,  11.06690 to  5.85034,  28.10945 nohead
set arrow from  7.75405,  11.06690 to  7.75405,  28.10945 nohead
set arrow from  9.10018,  11.06690 to  9.10018,  28.10945 nohead
set arrow from 10.44630,  11.06690 to 10.44630,  28.10945 nohead
set arrow from 12.09496,  11.06690 to 12.09496,  28.10945 nohead
set xtics ("G"  0.00000,"H"  1.90371,"P"  3.55237,"N"  4.50422,"G"  5.85034,"H"  7.75405,"N"  9.10018,"G" 10.44630,"P" 12.09496,"N" 13.04681)
 plot "niobium_band.dat"
