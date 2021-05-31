from skimage import io
import Load


path = '脊柱疾病智能诊断/train/data'
x, ytag = Load.DataLoad(path)
for itm in range(len(x)):
    imgname = x[itm].split('/')[-1].split('.jpg')[0]
    xml_file = open((r'VOCdevkit/VOC/Annotations' + '/' + imgname + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('\t<folder>' + 'VOC' + '</folder>\n')
    xml_file.write('\t<filename>' + imgname + '.jpg' + '</filename>\n')

    xml_file.write('\t<source>\n')
    xml_file.write('\t\t<database>' + 'The VOC Database' + '</database>\n')
    xml_file.write('\t\t<annotation>' + 'PASCAL VOC' + '</annotation>\n')
    xml_file.write('\t\t<image>flickr</image>\n')
    xml_file.write('\t\t<flickrid>325991873</flickrid>\n')
    xml_file.write('\t</source>\n')

    xml_file.write('\t<owner>\n')
    xml_file.write('\t\t<flickrid>archin</flickrid>\n')
    xml_file.write('\t\t<name>?</name>\n')
    xml_file.write('\t</owner>\n')

    img = io.imread(x[itm])
    width = img.shape[1]
    height = img.shape[0]
    xml_file.write('\t<size>\n')
    xml_file.write('\t\t<width>' + str(width) + '</width>\n')
    xml_file.write('\t\t<height>' + str(height) + '</height>\n')
    xml_file.write('\t\t<depth>3</depth>\n')
    xml_file.write('\t</size>\n')
    xml_file.write('\t<segmented>0</segmented>\n')

    gt = Load.getFrame(ytag[itm], bias=3)
    for img_each_label in gt:
        spt = img_each_label
        objnms = spt[5].split(',')
        for objname in objnms:
            xml_file.write('\t<object>\n')
            xml_file.write('\t\t<name>' + spt[4] + ',' + objname + '</name>\n')
            xml_file.write('\t\t<pose>Unspecified</pose>\n')
            xml_file.write('\t\t<truncated>0</truncated>\n')
            xml_file.write('\t\t<difficult>0</difficult>\n')
            xml_file.write('\t\t<bndbox>\n')
            xml_file.write('\t\t\t<xmin>' + str(int(spt[0])) + '</xmin>\n')
            xml_file.write('\t\t\t<ymin>' + str(int(spt[1])) + '</ymin>\n')
            xml_file.write('\t\t\t<xmax>' + str(int(spt[2])) + '</xmax>\n')
            xml_file.write('\t\t\t<ymax>' + str(int(spt[3])) + '</ymax>\n')
            xml_file.write('\t\t</bndbox>\n')
            xml_file.write('\t</object>\n')
    xml_file.write('</annotation>')

path = '脊柱疾病智能诊断/test/data'
x, ytag = Load.DataLoad(path)
for itm in range(len(x)):
    imgname = x[itm].split('/')[-1].split('.jpg')[0]
    xml_file = open((r'img/AnnotationsLabel' + '/' + imgname + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('\t<folder>' + 'VOC' + '</folder>\n')
    xml_file.write('\t<filename>' + imgname + '.jpg' + '</filename>\n')

    xml_file.write('\t<source>\n')
    xml_file.write('\t\t<database>' + 'The VOC Database' + '</database>\n')
    xml_file.write('\t\t<annotation>' + 'PASCAL VOC' + '</annotation>\n')
    xml_file.write('\t\t<image>flickr</image>\n')
    xml_file.write('\t\t<flickrid>325991873</flickrid>\n')
    xml_file.write('\t</source>\n')

    xml_file.write('\t<owner>\n')
    xml_file.write('\t\t<flickrid>archin</flickrid>\n')
    xml_file.write('\t\t<name>?</name>\n')
    xml_file.write('\t</owner>\n')

    img = io.imread(x[itm])
    width = img.shape[1]
    height = img.shape[0]
    xml_file.write('\t<size>\n')
    xml_file.write('\t\t<width>' + str(width) + '</width>\n')
    xml_file.write('\t\t<height>' + str(height) + '</height>\n')
    xml_file.write('\t\t<depth>3</depth>\n')
    xml_file.write('\t</size>\n')
    xml_file.write('\t<segmented>0</segmented>\n')

    gt = Load.getFrame(ytag[itm], bias=3)
    for img_each_label in gt:
        spt = img_each_label
        objnms = spt[5].split(',')
        for objname in objnms:
            xml_file.write('\t<object>\n')
            xml_file.write('\t\t<name>' + spt[4] + ',' + objname + '</name>\n')
            xml_file.write('\t\t<pose>Unspecified</pose>\n')
            xml_file.write('\t\t<truncated>0</truncated>\n')
            xml_file.write('\t\t<difficult>0</difficult>\n')
            xml_file.write('\t\t<bndbox>\n')
            xml_file.write('\t\t\t<xmin>' + str(int(spt[0])) + '</xmin>\n')
            xml_file.write('\t\t\t<ymin>' + str(int(spt[1])) + '</ymin>\n')
            xml_file.write('\t\t\t<xmax>' + str(int(spt[2])) + '</xmax>\n')
            xml_file.write('\t\t\t<ymax>' + str(int(spt[3])) + '</ymax>\n')
            xml_file.write('\t\t</bndbox>\n')
            xml_file.write('\t</object>\n')
    xml_file.write('</annotation>')
print('GetXml Finished!')

