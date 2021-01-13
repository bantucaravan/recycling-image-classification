import build_models



conv_base = build_models.get_model(<desc>, <opt>, <img_shape>)





# save inception resnet summary bc longer than longest allowed print in vscode
rec = []
def test(_str):
    rec.append(_str)
conv_base.summary(print_fn=test)
with open('../InceptionResNetV2 Summary.txt', 'w') as f:
    f.write('\n'.join(rec))



### get model graph
from tensorflow.keras.utils import plot_model, model_to_dot
dpi = 150
plot_model(conv_base, to_file=f'../InceptionResNetV2 graph {dpi}dpi.pdf', show_shapes=True, expand_nested=True, dpi=dpi)
# jpg not 96 dpi seems to output a zero byte file!!!!!


# using the `dot` command line tool (which pydot calls) setting dpi above 102 returns a zero 
# byte image for jpgs, it might be some internal dimension cut off thing, only with jpg tho (but jpg 
# is easiet ux when opening in gui)
dot = model_to_dot(conv_base, show_shapes=True, expand_nested=True, dpi=150, rankdir='TB')
dpi = 102
_format = 'jpg' # 'svg' # 'png' # 
to_file=f'../InceptionResNetV2 expanded graph {dpi}dpi.{_format}'
dot.write(to_file, format='jpg', prog=['dot', f'-Gdpi={dpi}'])




