rm channel_output.h5
cmake --build ./cmake-cache/ --target sls_chan_validation
./build/sls_chan_validation.exe
#python ../aerial-cuda-accelerated-ran/testBenches/chanModels/util/analysis_channel_stats.py ./channel_output.h5 --reference-json ../aerial-cuda-accelerated-ran/testBenches/chanModels/util/3gpp_calibration_phase1.json --calibration-phase 1
