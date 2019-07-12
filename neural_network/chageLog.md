
#2019/05/16  #
### change settings and data ###
* in a better, efficient way
* stacked_frames remain 8  skip_size changed to 4 from 5
* down-sample rate from 1/10 (skip 5 ,maxpooling 2) to 1/8 (skip 4 ,maxpooling 2)
* using delta and delta_delta


#2019/05/10  #
### rewrite attention mask ###
* in a better, efficient way

### add max pooling after second layer of LSTM ###
* the test result of adding maxpooling after 2nd LSTM is kinda good, around WER 0.3 without over fitting
* the results are in glenord CTC_test

### clean up code ###


#2019/04/27  #
###data reGenerate###
* split data into CSJ-APS and CSJ-SPS
* use test dataset picked by CSJ which declared in csj/DOC/asr.pdf
***
* data properties
    
    | name | value |
    | ---- | ----  |
    |'ori_filename'                  | file                                            | 
    |'core_noncore'                  | data_folder[:-1]                                |
    |'ori_sound'                     |f'ori_sound_{save_count}{count}.npy'             |
    |'fbank_feat'                    |f'fbank_feat_{save_count}{count}.npy'            |
    |'fbank_feat_mean_norm'          |f'fbank_feat_mean_norm_{save_count}{count}.npy'  |
    |'frame_length'                  |fbank_feat.shape[0]                              |
    |'PlainOrthographicTranscription'| text                                            |
    |'PhoneticTranscription'         | PhoneticTranscription                           |