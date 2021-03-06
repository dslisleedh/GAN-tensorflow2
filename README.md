# GAN-tensorflow2  

|Model|Discriminator(Critic) Loss|Generator Loss|
|--|--|--|
|<b>GAN</b>|![formula](https://render.githubusercontent.com/render/math?math=E[log(D(x))]%2bE[log(1%2dD(G(z)))])|![formula](https://render.githubusercontent.com/render/math?math=E[log(D(G(z)))])|
|<b>CGAN</b>|![formula](https://render.githubusercontent.com/render/math?math=E[log(D(x,y))]%2bE[log(1%2dD(G(z,y)))])|![formula](https://render.githubusercontent.com/render/math?math=E[log(D(G(z,y)))])|
|<b>LSGAN</b>|![formula](https://render.githubusercontent.com/render/math?math=\frac{1}{2}E[(D(x)%2d1)^2]%2b\frac{1}{2}E[(D(G(z)))^2])|![formula](https://render.githubusercontent.com/render/math?math=\frac{1}{2}E[(D(G(z))-1)^2])|
|<b>CycleGAN</b>|![formula](https://render.githubusercontent.com/render/math?math=\frac{1}{2}E[(D(x)%2d1)^2]%2b\frac{1}{2}E[(D(G(z)))^2])|<img src="https://render.githubusercontent.com/render/math?math=\begin{gathered} \frac{1}{2} E[(D(G(z) %2d 1)^2] \\ %2b \lambda Cycle(G,F) \\ %2b \frac{1}{2} \lambda Identity(G) \end{gathered}">|
|<b>WGAN</b>|<img src="https://render.githubusercontent.com/render/math?math=\begin{gathered} %2dE[D(x)] %2b E[D(G(z))] \\ W_d = Clip(W_d, %2dc, c) \\ c=0.01 \end{gathered}"> | ![formula](https://render.githubusercontent.com/render/math?math=-E[D(G(z))])|

