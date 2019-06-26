# Code Progress Sequence



**< for sentence : bookshelf stories >**

> 1. Load Image
>
> 2. preprocess images
>
> 3. bring mid coordinates
>
> 4. get 'bgr' from mid-coordinates
>
>    >**< for sentence : divide position coordinates (rect) >**
>    >
>    >1. get a single x, y coordinate
>    >2. convert float to int
>    >3. get 'bgr' from the coordinate
>
> 5. get book's height
>
> 6. get position value
>
> 7. ann : SOM
>
>    > 1. preparing data
>    >
>    > 2. build som-network model  /  train model
>    >    : 고차원의 'r','g','b','height' 값을 거리를 기준으로 2D로 clustering 한다.
>    >
>    > 3. visualization
>    >
>    >    >1. 2D view
>    >    >2. U-matrix plot
>    >    >3. hitmap view
>    >
>    > 4. K-means clustering on the SOM-grid
>    >
>    > 5. show graphics
>    >
>    > 6. labeling coordinates
>    >
>    > 7. set 'current index', 'target index'
>
> 8. rearrange books ***
>
>    > 1. get an empty bookshelf
>    >
>    > 2. define variables
>    >
>    > 3. paste books on new bookshelf
>    >
>    >    > **< for sentence : divide shelf >**
>    >    >
>    >    > 1. get bookshelf's paste-target coord
>    >    >
>    >    >    > **<for sentence : number of books>**
>    >    >    >
>    >    >    > 1. select coordinate of book to paste
>    >    >    > 2. slicing single book in image
>    >    >    > 3. paste book image to bookshelf
>
> 9. post-process image
>
> 10. show and write image
>
> 11. end the session