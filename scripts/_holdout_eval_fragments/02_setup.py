%cd /content
![ -d CT-CLIP ] || git clone https://github.com/ibrahimethemhamamci/CT-CLIP.git
%cd /content/CT-CLIP
!pip install -q -e transformer_maskgit
!pip install -q -e CT_CLIP
!pip install -q scikit-learn pandas matplotlib tqdm
import sys
for p in ['/content/CT-CLIP/CT_CLIP', '/content/CT-CLIP/transformer_maskgit']:
    if p not in sys.path:
        sys.path.insert(0, p)
%cd /content
![ -d 3dCT ] || git clone https://github.com/nprakash1/3dCT.git
%cd /content/3dCT
!git pull --ff-only || true
import ct_clip, transformer_maskgit
print('CT-CLIP import OK')
