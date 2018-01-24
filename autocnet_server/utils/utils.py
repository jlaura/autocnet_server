import os

def create_output_path(ds, outdir):
    image_name = os.path.basename(ds.file_name)
    image_path = os.path.dirname(ds.file_name)

    if outdir is None:
        outh5 = os.path.join(image_path, image_name + '_kps.h5')
    else:
        outh5 = os.path.join(outdir, image_name + '_kps.h5')

    return outh5
