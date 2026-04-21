set style data dots
set nokey
set xrange [0: 3.79495]
set yrange [ -6.73511 : 17.31446]
set arrow from  1.00191,  -6.73511 to  1.00191,  17.31446 nohead
set arrow from  2.15883,  -6.73511 to  2.15883,  17.31446 nohead
set arrow from  2.56786,  -6.73511 to  2.56786,  17.31446 nohead
set xtics ("L"  0.00000,"G"  1.00191,"X"  2.15883,"K"  2.56786,"G"  3.79495)
 plot "silicon_band.dat"
