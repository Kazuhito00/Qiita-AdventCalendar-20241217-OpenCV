# PyScript-Private
python -m http.server 8000
<br>
http://localhost:8000
<br>
opencv_zoo
<br>

---
* [HTMLファイル内にPythonのコードを記述するツール「PyScript」を使ってみた](https://zenn.dev/torakm/articles/92fd244974efd6#todo-%E3%82%A2%E3%83%97%E3%83%AA%E3%81%AE%E3%82%B3%E3%83%BC%E3%83%89)
* [ME35 OpenCV Playground](https://chrisrogers.pyscriptapps.com/me35-camera/latest/)
    * ```python
      cv2_image = cv2.cvtColor(np.array(cam.raw_image), cv2.COLOR_RGB2BGR)
      b,g,r = cv2.split(cv2_image)
      grey = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2GRAY)
      cam.show(grey)  # shows any cv2 image in the same spot on the webpage (third image)
      image3 = Image.fromarray(grey)
      display(Image.fromarray(r),Image.fromarray(g),Image.fromarray(b))
      
      #cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
      color = ('b','g','r')
      for i,col in enumerate(color):
          histr = cv2.calcHist([cv2_image],[i],None,[256],[0,256])
          plt.plot(histr,color = col)  # add the different histograms to the plot
          plt.xlim([0,256])  # define x axis length (cuts off some of the picture)
      
      plt.imshow(r)  # puts red image in the background
      display(plt)  #shows it
      ```
