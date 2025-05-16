from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt

def display_middle_slice(nifti_path: str, plane: str = "axial"):
    """
    it load a NIfTI file and display its middle slice.
    
    Args:
        nifti_path: Path to a .nii or .nii.gz file.
        plane:      Which plane to view: "axial", "coronal", or "sagittal".
        axial is slice in the z-axis,
        coronal is slice in the y-axis,
        sagittal is slice in the x-axis.
    """
    p = Path(nifti_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    
    # Load volume
    img = nib.load(str(p))
    vol = img.get_fdata()
    
    # Choose slice index
    if plane == "axial":
        idx = vol.shape[2] // 2
        slice_img = vol[:, :, idx].T
    elif plane == "coronal":
        idx = vol.shape[1] // 2
        slice_img = vol[:, idx, :].T
    elif plane == "sagittal":
        idx = vol.shape[0] // 2
        slice_img = vol[idx, :, :].T
    else:
        raise ValueError("Plane must be 'axial', 'coronal', or 'sagittal'")
    
    # Plot
    plt.figure(figsize=(6,6))
    plt.imshow(slice_img, cmap="gray", origin="lower")
    plt.title(f"{p.name} — {plane.capitalize()} slice {idx}")
    plt.axis("off")
    plt.show()
