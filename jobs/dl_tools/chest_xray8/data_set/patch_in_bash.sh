for i in {0..3}; do
    python jobs/dl_tools/chest_xray8/Patch_maker.py --i $i &
done