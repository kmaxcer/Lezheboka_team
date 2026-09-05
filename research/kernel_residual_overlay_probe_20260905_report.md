# Kernel residual overlay probe

Residual = weighted temporal neighbor estimate minus robust-base prediction; evaluated on released old GT.

kernel    h  alpha     rmse  base_rmse    delta
   tri 0.05   0.05 0.061841   0.061609 0.000232
   tri 0.05   0.10 0.062210   0.061609 0.000601
   tri 0.05   0.15 0.062714   0.061609 0.001105
   tri 0.10   0.05 0.061944   0.061609 0.000335
   tri 0.10   0.10 0.062443   0.061609 0.000834
   tri 0.10   0.15 0.063101   0.061609 0.001492
   tri 0.20   0.05 0.062187   0.061609 0.000578
   tri 0.20   0.10 0.063110   0.061609 0.001501
   tri 0.20   0.15 0.064364   0.061609 0.002755
 gauss 0.05   0.05 0.061988   0.061609 0.000379
 gauss 0.05   0.10 0.062553   0.061609 0.000944
 gauss 0.05   0.15 0.063301   0.061609 0.001692
 gauss 0.10   0.05 0.062252   0.061609 0.000642
 gauss 0.10   0.10 0.063304   0.061609 0.001695
 gauss 0.10   0.15 0.064745   0.061609 0.003136
 gauss 0.20   0.05 0.062544   0.061609 0.000934
 gauss 0.20   0.10 0.064304   0.061609 0.002695
 gauss 0.20   0.15 0.066825   0.061609 0.005216

No candidate materialized unless independent masks improve.
