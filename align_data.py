import nibabel as nib
import numpy as np
from Wehbe_loader import words, time, meta
from nilearn import plotting
from nilearn.image import index_img, concat_imgs
import pandas as pd

time_data = time

words = words[0]

if __name__ == '__main__':

    def align_words_with_fmri(subject):

        # Load the .npy files
        time_fmri = np.load('Wehbe_data/time_fmri.npy')
        time_words_fmri = np.load('Wehbe_data/time_words_fmri.npy')
        words_fmri = np.load('Wehbe_data/words_fmri.npy')

        # Load the NIfTI file
        path = f'data/{subject}_almost_preprocessed_volume.nii'
        nifti_image = nib.load(path)

        # Get the data from the NIfTI file
        preprocessed_volume = nifti_image.get_fdata()

        # Get the affine matrix from the NIfTI file
        affine = nifti_image.affine

        # Get the number of images and words
        num_images = preprocessed_volume.shape[3]

        num_words = len(words)

        # Ensure that the number of images matches the number of time points
        assert num_images == len(time_data), "The number of images should match the number of time points in time_data."

        # Align words with images based on time intervals
        aligned_words = []
        aligned_fmri_data = []
        aligned_data = []


        niimgs = []
        for i in range(num_images):
            image_time = float(time_fmri[i])
            start_time = image_time - 6
            end_time = image_time - 4
            words_at_time = [words_fmri[k] for k in range(len(time_words_fmri)) if start_time <= time_words_fmri[k] < end_time]
            if len(words_at_time) == 4:
                fmri_data_segments = index_img(nifti_image, i)
                niimgs.append(fmri_data_segments)
                aligned_data.append({'words': words_at_time, 'fmri_data': fmri_data_segments})

        # Slice only relevant volumes
        preprocessed_volume_img = concat_imgs(niimgs)
        print(preprocessed_volume_img.shape)

        # Save the preprocessed volume
        output_file = f'{subject}_preprocessed_volume.nii'
        nib.save(preprocessed_volume_img, output_file)
        print(f'Saved preprocessed data for {subject} to {output_file}')

    align_words_with_fmri()

    

    """    # Convert aligned words and fMRI data to numpy arrays
        #aligned_words_array = np.array(aligned_words, dtype=object)  # Use dtype=object for arrays of lists
        #aligned_fmri_data_array = np.array(aligned_fmri_data, dtype=object)

        # Convert aligned data to a numpy array
        aligned_data_array = np.array(aligned_data, dtype=object)

        file_name_npz = 'aligned_data_{subject}.npz'

        # Save the aligned data to an .npz file
        np.savez(file_name_npz, aligned_data=aligned_data_array)

        file_name_csv = 'aligned_data_{subject}.csv'
        
        # Convert aligned data to a pandas DataFrame
        aligned_df = pd.DataFrame(aligned_data)

        # Save the DataFrame to a .csv file using pandas
        aligned_df.to_csv(file_name_csv)
    
    align_words_with_fmri('subject_4')"""
