from Setup_Path import ANNOTATION_PATH

labels = [{'name':'Hello', 'id':1},
          {'name':'ILoveYou', 'id':2},
          {'name':'Yes', 'id':3},
          {'name':'No', 'id':4},
          {'name':'Thanks', 'id':5},
         ]

with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')