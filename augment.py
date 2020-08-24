import Augmentor

p = Augmentor.Pipeline('/root/torchbox/dataset/data_image/Pneumonia')
#p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
#p.shear(probability=1,max_shear_right=5,max_shear_left=5)
#p.flip_left_right(probability=1)
#p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
p.resize(probability=1.0, width=512, height=512)
p.process()
