* Added arguement in finetuning_args.py to introduce two variables to choose the type of training loss to use
    * use_only_motionvector_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the motion vector loss."},
        )
    * use_motionvector_crossentropy_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the motion vector and cross-entropy loss."},
        )
    * These can be invoked through cli now (I am guessing)
    * Now we will use these in the `trainer.py` to invoke the custom `self.compute_loss_func()`.
        * Created two new functions in the `trainer_utils.py` for calculating these corresponding losses.
            * TODO : Write code in these functions, keeping empty as of now.

* Added `train_on_video_tokens` in the `DataArguements` in `data_args.py`
    * This will be used in `supervised.py` at line 63 where 



* The dimensions of the input is 768 x n_images.
    * The dimensions of the motion vectors would be `[2(x,y) * num_patches] * n_images`
        * where `num_pathces = (img_width * img_height) / (x_patch * y_patch)`

    * TODO : Need to add a linear layer to convert the visual tokens dimensions being output to the motion vector dimension.



```
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
```


# THE MOST IMPORTANT THING
* `llamafactory/data/processor/supervised.py` is where we have to add the motion vectors in `def _encode_data_example()` - in `elif self.data_args.train_on_video_tokens` (line 70)
    *Just need to change these for varun sir's experiment as well

* ARE THE VISION TOKENS ALSO GENERATED SEQUENTIALLY?



### IPMORTANT INFO
* MAJOR CHANGE IS TO BE DONE IN `supervised.py` IN `SupervisedDatasetProcessor`'s `_encode_data_example()` function, where we have to change the source_ids, target_ids, 
* WORKFLOW : 
    0. Starts from `train.py` or `clil.py` where `run_exp()` is called
    1. This further calls the `_training_function()` in `llamafactory/train/tuner.py` which calls `run_sft()`, which is present in `train/sft/workflow.py`
    2. `llamafactory/train/sft/workflow.py` : Here is creates the `DataCollatorWith4DAttentionMask`
    3. The `DataCollatorWith4DAttentionMask` is in `llamafactory/data/collator.py` and is a subclass of `MultiModalDataCollatorForSeq2Seq` (which itself is a subclass of `DataCollatorForSeq2Seq`): The __call__ fuction of the parent class call the `get_mm_inputs()` from the mm_plugin of the template which is `Qwen2VLPlugin`
        * Not able to print anything in the call function of the collator function.
    4. The `Qwen2VLPlugin` is present in `llamafactory/data/mm_plugin.py`, which is asubclass of `BasePlugin` (which itself is a subclass of `MMPluginMixin`)
        * The `get_mm_inputs()` is present in `BasePlugin` which further calls the `self._get_mm_inputs()`, which is overriden by `Qwen2VLPlugin` over the one in `MMPluginMixin`




* Can setup the default fps in the `_regularize_videos()` functions for all the plugins in `mm_plugin.py` or somehow find how to change it in kwargs.
    * 

* 
    ```
Using template: Template(`format_user=StringFormatter(slots=['<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n']`, tool_format=None), `format_assistant=StringFormatter(slots=['{{content}}<|im_end|>\n']`, tool_format=None), `format_system=StringFormatter(slots=['<|im_start|>system\n{{content}}<|im_end|>\n']`, tool_format=None), format_function=FunctionFormatter(slots=['{{content}}<|im_end|>\n'], tool_format='qwen'), format_observation=StringFormatter(slots=['<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n'], tool_format=None), format_tools=ToolFormatter(slots=[], tool_format='qwen'), format_prefix=EmptyFormatter(slots=[], tool_format=None), default_system='You are a helpful assistant.', stop_words=['<|im_end|>'], thought_words=('<think>\n', '\n</think>\n\n'), efficient_eos=False, replace_eos=True, replace_jinja_template=False, enable_thinking=False, `mm_plugin=Qwen2VLPlugin(image_token='<|image_pad|>', video_token='<|video_pad|>'`, audio_token=None, expand_mm_tokens=True, start_token='<|vision_start|>', end_token='<|vision_end|>'))
    ```

### DOUBTS
* The inputs are being passed as 
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|> .....<|video_pad|><|video_pad|> .....<|video_pad|><|vision_end|> Explain the action happening the video.<|im_end|>
<|im_start|>assistant
Spinning cube that quickly stops spinning<|im_end|>
```
* The labels are
```
<IGNORE_INDEX><IGNORE_INDEX>...Spinning cube that quickly stops spinning<|im_end|>
```


Is this the correct way to do it?



### TODO

* Add code to the new loss functions for calculating the motion vector loss and the motion vector + cross entropy loss
* Make changes in the `prediction_step()` function in `trainer.py` file to not mask the inputs of the visual vectors
    * The current code removes all the parts of the prompt which has both text and visual tokens and then returns just the newly generated tokens
    * How do we pass the length of the visual tokens so we can add those vectors to the generated_tokens 
* Need to add a linear layer to convert the visual tokens dimensions being output to the motion vector dimension.
* Change the masking strategy in the dataloader



### What do the variables contain
* `inputs` : Dictionary of two keys : `input_ids` and `labels`.
* `dataset` : Dictionary of two keys : `train_dataset` and `eval_dataset`
* 


### Things to take care of
* Qwen2-VL uses dynamic resolution; ensure your motion grid is computed after any resizing fed to the vision encoder (so patch grids match).
* 