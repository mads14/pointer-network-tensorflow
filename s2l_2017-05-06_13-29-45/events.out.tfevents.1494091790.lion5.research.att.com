       £K"	  CÖAbrain.Event:2u}	\>     £fg	±2CÖA"±
f
PlaceholderPlaceholder*
shape: *
dtype0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
[
Placeholder_1Placeholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape: 
X

early_stopPlaceholder*#
_output_shapes
:’’’’’’’’’*
shape: *
dtype0
ė
random_queue_testRandomShuffleQueueV2*#
shapes
:	::*

seed{*
shared_name *
min_after_dequeueč*
capacityč
*
	container *
seed2ļ*
_output_shapes
: *
component_types
2

random_queue_test_enqueueQueueEnqueueV2random_queue_testPlaceholderPlaceholder_1
early_stop*
Tcomponents
2*

timeout_ms’’’’’’’’’
¢
random_queue_test_DequeueQueueDequeueV2random_queue_test*

timeout_ms’’’’’’’’’*+
_output_shapes
:	::*
component_types
2
U
batch_and_pad/ConstConst*
value	B
 Z*
_output_shapes
: *
dtype0

Ä
 batch_and_pad/padding_fifo_queuePaddingFIFOQueueV2*#
shapes
:	::*
	container *
_output_shapes
: *
component_types
2*
capacityč
*
shared_name 
p
batch_and_pad/cond/SwitchSwitchbatch_and_pad/Constbatch_and_pad/Const*
T0
*
_output_shapes
: : 
e
batch_and_pad/cond/switch_tIdentitybatch_and_pad/cond/Switch:1*
T0
*
_output_shapes
: 
c
batch_and_pad/cond/switch_fIdentitybatch_and_pad/cond/Switch*
T0
*
_output_shapes
: 
\
batch_and_pad/cond/pred_idIdentitybatch_and_pad/Const*
_output_shapes
: *
T0

Ō
4batch_and_pad/cond/padding_fifo_queue_enqueue/SwitchSwitch batch_and_pad/padding_fifo_queuebatch_and_pad/cond/pred_id*
_output_shapes
: : *3
_class)
'%loc:@batch_and_pad/padding_fifo_queue*
T0
Ś
6batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_1Switchrandom_queue_test_Dequeuebatch_and_pad/cond/pred_id*
T0*,
_class"
 loc:@random_queue_test_Dequeue**
_output_shapes
:	:	
Ņ
6batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_2Switchrandom_queue_test_Dequeue:1batch_and_pad/cond/pred_id*,
_class"
 loc:@random_queue_test_Dequeue* 
_output_shapes
::*
T0
Ņ
6batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_3Switchrandom_queue_test_Dequeue:2batch_and_pad/cond/pred_id* 
_output_shapes
::*,
_class"
 loc:@random_queue_test_Dequeue*
T0
Ų
-batch_and_pad/cond/padding_fifo_queue_enqueueQueueEnqueueV26batch_and_pad/cond/padding_fifo_queue_enqueue/Switch:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_1:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_2:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_3:1*
Tcomponents
2*

timeout_ms’’’’’’’’’
Ļ
%batch_and_pad/cond/control_dependencyIdentitybatch_and_pad/cond/switch_t.^batch_and_pad/cond/padding_fifo_queue_enqueue*.
_class$
" loc:@batch_and_pad/cond/switch_t*
_output_shapes
: *
T0

=
batch_and_pad/cond/NoOpNoOp^batch_and_pad/cond/switch_f
»
'batch_and_pad/cond/control_dependency_1Identitybatch_and_pad/cond/switch_f^batch_and_pad/cond/NoOp*
_output_shapes
: *.
_class$
" loc:@batch_and_pad/cond/switch_f*
T0


batch_and_pad/cond/MergeMerge'batch_and_pad/cond/control_dependency_1%batch_and_pad/cond/control_dependency*
T0
*
N*
_output_shapes
: : 
w
&batch_and_pad/padding_fifo_queue_CloseQueueCloseV2 batch_and_pad/padding_fifo_queue*
cancel_pending_enqueues( 
y
(batch_and_pad/padding_fifo_queue_Close_1QueueCloseV2 batch_and_pad/padding_fifo_queue*
cancel_pending_enqueues(
n
%batch_and_pad/padding_fifo_queue_SizeQueueSizeV2 batch_and_pad/padding_fifo_queue*
_output_shapes
: 
q
batch_and_pad/CastCast%batch_and_pad/padding_fifo_queue_Size*

SrcT0*
_output_shapes
: *

DstT0
X
batch_and_pad/mul/yConst*
valueB
 *i=:*
_output_shapes
: *
dtype0
b
batch_and_pad/mulMulbatch_and_pad/Castbatch_and_pad/mul/y*
_output_shapes
: *
T0

(batch_and_pad/fraction_of_1384_full/tagsConst*
dtype0*
_output_shapes
: *4
value+B) B#batch_and_pad/fraction_of_1384_full

#batch_and_pad/fraction_of_1384_fullScalarSummary(batch_and_pad/fraction_of_1384_full/tagsbatch_and_pad/mul*
T0*
_output_shapes
: 
R
batch_and_pad/nConst*
_output_shapes
: *
dtype0*
value
B :
É
batch_and_padQueueDequeueManyV2 batch_and_pad/padding_fifo_queuebatch_and_pad/n*:
_output_shapes(
&::	:	*
component_types
2*

timeout_ms’’’’’’’’’
h
Placeholder_2Placeholder*
dtype0*
shape: *0
_output_shapes
:’’’’’’’’’’’’’’’’’’
[
Placeholder_3Placeholder*
shape: *
dtype0*#
_output_shapes
:’’’’’’’’’
Z
early_stop_1Placeholder*
dtype0*
shape: *#
_output_shapes
:’’’’’’’’’
ė
random_queue_trainRandomShuffleQueueV2*#
shapes
:	::*

seed{*
shared_name *
min_after_dequeueč*
capacityč
*
	container *
seed2{*
_output_shapes
: *
component_types
2

random_queue_train_enqueueQueueEnqueueV2random_queue_trainPlaceholder_2Placeholder_3early_stop_1*
Tcomponents
2*

timeout_ms’’’’’’’’’
¤
random_queue_train_DequeueQueueDequeueV2random_queue_train*

timeout_ms’’’’’’’’’*+
_output_shapes
:	::*
component_types
2
W
batch_and_pad_1/ConstConst*
value	B
 Z*
_output_shapes
: *
dtype0

Ę
"batch_and_pad_1/padding_fifo_queuePaddingFIFOQueueV2*#
shapes
:	::*
	container *
shared_name *
_output_shapes
: *
component_types
2*
capacityč

v
batch_and_pad_1/cond/SwitchSwitchbatch_and_pad_1/Constbatch_and_pad_1/Const*
_output_shapes
: : *
T0

i
batch_and_pad_1/cond/switch_tIdentitybatch_and_pad_1/cond/Switch:1*
_output_shapes
: *
T0

g
batch_and_pad_1/cond/switch_fIdentitybatch_and_pad_1/cond/Switch*
T0
*
_output_shapes
: 
`
batch_and_pad_1/cond/pred_idIdentitybatch_and_pad_1/Const*
_output_shapes
: *
T0

Ü
6batch_and_pad_1/cond/padding_fifo_queue_enqueue/SwitchSwitch"batch_and_pad_1/padding_fifo_queuebatch_and_pad_1/cond/pred_id*
T0*5
_class+
)'loc:@batch_and_pad_1/padding_fifo_queue*
_output_shapes
: : 
ą
8batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_1Switchrandom_queue_train_Dequeuebatch_and_pad_1/cond/pred_id*
T0**
_output_shapes
:	:	*-
_class#
!loc:@random_queue_train_Dequeue
Ų
8batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_2Switchrandom_queue_train_Dequeue:1batch_and_pad_1/cond/pred_id*
T0*-
_class#
!loc:@random_queue_train_Dequeue* 
_output_shapes
::
Ų
8batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_3Switchrandom_queue_train_Dequeue:2batch_and_pad_1/cond/pred_id* 
_output_shapes
::*-
_class#
!loc:@random_queue_train_Dequeue*
T0
ā
/batch_and_pad_1/cond/padding_fifo_queue_enqueueQueueEnqueueV28batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch:1:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_1:1:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_2:1:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_3:1*
Tcomponents
2*

timeout_ms’’’’’’’’’
×
'batch_and_pad_1/cond/control_dependencyIdentitybatch_and_pad_1/cond/switch_t0^batch_and_pad_1/cond/padding_fifo_queue_enqueue*
_output_shapes
: *0
_class&
$"loc:@batch_and_pad_1/cond/switch_t*
T0

A
batch_and_pad_1/cond/NoOpNoOp^batch_and_pad_1/cond/switch_f
Ć
)batch_and_pad_1/cond/control_dependency_1Identitybatch_and_pad_1/cond/switch_f^batch_and_pad_1/cond/NoOp*
T0
*0
_class&
$"loc:@batch_and_pad_1/cond/switch_f*
_output_shapes
: 
£
batch_and_pad_1/cond/MergeMerge)batch_and_pad_1/cond/control_dependency_1'batch_and_pad_1/cond/control_dependency*
T0
*
N*
_output_shapes
: : 
{
(batch_and_pad_1/padding_fifo_queue_CloseQueueCloseV2"batch_and_pad_1/padding_fifo_queue*
cancel_pending_enqueues( 
}
*batch_and_pad_1/padding_fifo_queue_Close_1QueueCloseV2"batch_and_pad_1/padding_fifo_queue*
cancel_pending_enqueues(
r
'batch_and_pad_1/padding_fifo_queue_SizeQueueSizeV2"batch_and_pad_1/padding_fifo_queue*
_output_shapes
: 
u
batch_and_pad_1/CastCast'batch_and_pad_1/padding_fifo_queue_Size*
_output_shapes
: *

DstT0*

SrcT0
Z
batch_and_pad_1/mul/yConst*
dtype0*
_output_shapes
: *
valueB
 *i=:
h
batch_and_pad_1/mulMulbatch_and_pad_1/Castbatch_and_pad_1/mul/y*
T0*
_output_shapes
: 

*batch_and_pad_1/fraction_of_1384_full/tagsConst*
_output_shapes
: *
dtype0*6
value-B+ B%batch_and_pad_1/fraction_of_1384_full

%batch_and_pad_1/fraction_of_1384_fullScalarSummary*batch_and_pad_1/fraction_of_1384_full/tagsbatch_and_pad_1/mul*
T0*
_output_shapes
: 
T
batch_and_pad_1/nConst*
value
B :*
_output_shapes
: *
dtype0
Ļ
batch_and_pad_1QueueDequeueManyV2"batch_and_pad_1/padding_fifo_queuebatch_and_pad_1/n*:
_output_shapes(
&::	:	*
component_types
2*

timeout_ms’’’’’’’’’
G
ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
^
is_trainingPlaceholderWithDefaultConst*
shape: *
dtype0
*
_output_shapes
: 
R
cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

I
cond/switch_tIdentitycond/Switch:1*
T0
*
_output_shapes
: 
G
cond/switch_fIdentitycond/Switch*
T0
*
_output_shapes
: 
F
cond/pred_idIdentityis_training*
_output_shapes
: *
T0


cond/Switch_1Switchbatch_and_pad_1cond/pred_id*4
_output_shapes"
 ::*"
_class
loc:@batch_and_pad_1*
T0

cond/Switch_2Switchbatch_and_pad_1:1cond/pred_id**
_output_shapes
:	:	*"
_class
loc:@batch_and_pad_1*
T0

cond/Switch_3Switchbatch_and_pad_1:2cond/pred_id*
T0**
_output_shapes
:	:	*"
_class
loc:@batch_and_pad_1

cond/Switch_4Switchbatch_and_padcond/pred_id*
T0* 
_class
loc:@batch_and_pad*4
_output_shapes"
 ::

cond/Switch_5Switchbatch_and_pad:1cond/pred_id**
_output_shapes
:	:	* 
_class
loc:@batch_and_pad*
T0

cond/Switch_6Switchbatch_and_pad:2cond/pred_id*
T0* 
_class
loc:@batch_and_pad**
_output_shapes
:	:	
m

cond/MergeMergecond/Switch_4cond/Switch_1:1*
N*
T0*&
_output_shapes
:: 
j
cond/Merge_1Mergecond/Switch_5cond/Switch_2:1*!
_output_shapes
:	: *
T0*
N
j
cond/Merge_2Mergecond/Switch_6cond/Switch_3:1*
T0*
N*!
_output_shapes
:	: 
j
cond/Merge_3Mergecond/Switch_6cond/Switch_3:1*!
_output_shapes
:	: *
N*
T0
`
Reshape/shapeConst*
valueB:
’’’’’’’’’*
_output_shapes
:*
dtype0
c
ReshapeReshapecond/Merge_2Reshape/shape*
T0*
_output_shapes	
:*
Tshape0
X
Variable/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
l
Variable
VariableV2*
shared_name *
dtype0*
shape: *
_output_shapes
: *
	container 
¢
Variable/AssignAssignVariableVariable/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@Variable
a
Variable/readIdentityVariable*
T0*
_output_shapes
: *
_class
loc:@Variable
”
,input_embed/Initializer/random_uniform/shapeConst*
_class
loc:@input_embed*!
valueB"         *
dtype0*
_output_shapes
:

*input_embed/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
_class
loc:@input_embed*
valueB
 *
×£½

*input_embed/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
_class
loc:@input_embed*
valueB
 *
×£=
ē
4input_embed/Initializer/random_uniform/RandomUniformRandomUniform,input_embed/Initializer/random_uniform/shape*
_class
loc:@input_embed*#
_output_shapes
:*
T0*
dtype0*
seed2W*

seed{
Ź
*input_embed/Initializer/random_uniform/subSub*input_embed/Initializer/random_uniform/max*input_embed/Initializer/random_uniform/min*
_class
loc:@input_embed*
_output_shapes
: *
T0
į
*input_embed/Initializer/random_uniform/mulMul4input_embed/Initializer/random_uniform/RandomUniform*input_embed/Initializer/random_uniform/sub*
T0*#
_output_shapes
:*
_class
loc:@input_embed
Ó
&input_embed/Initializer/random_uniformAdd*input_embed/Initializer/random_uniform/mul*input_embed/Initializer/random_uniform/min*
T0*
_class
loc:@input_embed*#
_output_shapes
:
©
input_embed
VariableV2*
shared_name *
_class
loc:@input_embed*
	container *
shape:*
dtype0*#
_output_shapes
:
Č
input_embed/AssignAssigninput_embed&input_embed/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@input_embed*
validate_shape(*#
_output_shapes
:
w
input_embed/readIdentityinput_embed*#
_output_shapes
:*
_class
loc:@input_embed*
T0
_
encoder/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

encoder/conv1d/ExpandDims
ExpandDims
cond/Mergeencoder/conv1d/ExpandDims/dim*(
_output_shapes
:*
T0*

Tdim0
a
encoder/conv1d/ExpandDims_1/dimConst*
value	B : *
_output_shapes
: *
dtype0

encoder/conv1d/ExpandDims_1
ExpandDimsinput_embed/readencoder/conv1d/ExpandDims_1/dim*'
_output_shapes
:*
T0*

Tdim0
ć
encoder/conv1d/Conv2DConv2Dencoder/conv1d/ExpandDimsencoder/conv1d/ExpandDims_1*)
_output_shapes
:*
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingVALID

encoder/conv1d/SqueezeSqueezeencoder/conv1d/Conv2D*
T0*%
_output_shapes
:*
squeeze_dims

Z
ShapeConst*
dtype0*
_output_shapes
:*!
valueB"         
]
strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ł
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
¬
)encoder/initial_state_0/Initializer/ConstConst*
dtype0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0*
valueB	*    
¹
encoder/initial_state_0
VariableV2**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
shape:	*
dtype0*
shared_name *
	container 
ė
encoder/initial_state_0/AssignAssignencoder/initial_state_0)encoder/initial_state_0/Initializer/Const*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(

encoder/initial_state_0/readIdentityencoder/initial_state_0*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0
m
+encoder_1/initial_state_0_tiled/multiples/1Const*
value	B :*
_output_shapes
: *
dtype0
§
)encoder_1/initial_state_0_tiled/multiplesPackstrided_slice+encoder_1/initial_state_0_tiled/multiples/1*
_output_shapes
:*
N*

axis *
T0
µ
encoder_1/initial_state_0_tiledTileencoder/initial_state_0/read)encoder_1/initial_state_0_tiled/multiples*(
_output_shapes
:’’’’’’’’’*
T0*

Tmultiples0
¬
)encoder/initial_state_1/Initializer/ConstConst**
_class 
loc:@encoder/initial_state_1*
valueB	*    *
dtype0*
_output_shapes
:	
¹
encoder/initial_state_1
VariableV2*
shared_name **
_class 
loc:@encoder/initial_state_1*
	container *
shape:	*
dtype0*
_output_shapes
:	
ė
encoder/initial_state_1/AssignAssignencoder/initial_state_1)encoder/initial_state_1/Initializer/Const*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_1*
T0*
use_locking(

encoder/initial_state_1/readIdentityencoder/initial_state_1**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	*
T0
m
+encoder_1/initial_state_1_tiled/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
§
)encoder_1/initial_state_1_tiled/multiplesPackstrided_slice+encoder_1/initial_state_1_tiled/multiples/1*
T0*

axis *
N*
_output_shapes
:
µ
encoder_1/initial_state_1_tiledTileencoder/initial_state_1/read)encoder_1/initial_state_1_tiled/multiples*

Tmultiples0*
T0*(
_output_shapes
:’’’’’’’’’
¬
)encoder/initial_state_2/Initializer/ConstConst**
_class 
loc:@encoder/initial_state_2*
valueB	*    *
dtype0*
_output_shapes
:	
¹
encoder/initial_state_2
VariableV2*
	container *
shared_name *
dtype0*
shape:	*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2
ė
encoder/initial_state_2/AssignAssignencoder/initial_state_2)encoder/initial_state_2/Initializer/Const*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_2*
validate_shape(*
_output_shapes
:	

encoder/initial_state_2/readIdentityencoder/initial_state_2*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2*
T0
m
+encoder_1/initial_state_2_tiled/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
§
)encoder_1/initial_state_2_tiled/multiplesPackstrided_slice+encoder_1/initial_state_2_tiled/multiples/1*
N*
T0*
_output_shapes
:*

axis 
µ
encoder_1/initial_state_2_tiledTileencoder/initial_state_2/read)encoder_1/initial_state_2_tiled/multiples*(
_output_shapes
:’’’’’’’’’*
T0*

Tmultiples0
¬
)encoder/initial_state_3/Initializer/ConstConst**
_class 
loc:@encoder/initial_state_3*
valueB	*    *
_output_shapes
:	*
dtype0
¹
encoder/initial_state_3
VariableV2*
_output_shapes
:	*
dtype0*
shape:	*
	container **
_class 
loc:@encoder/initial_state_3*
shared_name 
ė
encoder/initial_state_3/AssignAssignencoder/initial_state_3)encoder/initial_state_3/Initializer/Const*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	

encoder/initial_state_3/readIdentityencoder/initial_state_3*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3
m
+encoder_1/initial_state_3_tiled/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
§
)encoder_1/initial_state_3_tiled/multiplesPackstrided_slice+encoder_1/initial_state_3_tiled/multiples/1*
T0*

axis *
N*
_output_shapes
:
µ
encoder_1/initial_state_3_tiledTileencoder/initial_state_3/read)encoder_1/initial_state_3_tiled/multiples*

Tmultiples0*
T0*(
_output_shapes
:’’’’’’’’’
m
encoder_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          

encoder_1/transpose	Transposeencoder/conv1d/Squeezeencoder_1/transpose/perm*
Tperm0*%
_output_shapes
:*
T0
T
encoder_1/sequence_lengthIdentityReshape*
T0*
_output_shapes	
:
h
encoder_1/rnn/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         
k
!encoder_1/rnn/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
m
#encoder_1/rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
m
#encoder_1/rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
æ
encoder_1/rnn/strided_sliceStridedSliceencoder_1/rnn/Shape!encoder_1/rnn/strided_slice/stack#encoder_1/rnn/strided_slice/stack_1#encoder_1/rnn/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0*
end_mask *
new_axis_mask *

begin_mask *
ellipsis_mask 
m
#encoder_1/rnn/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
o
%encoder_1/rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
o
%encoder_1/rnn/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ē
encoder_1/rnn/strided_slice_1StridedSliceencoder_1/rnn/Shape#encoder_1/rnn/strided_slice_1/stack%encoder_1/rnn/strided_slice_1/stack_1%encoder_1/rnn/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
`
encoder_1/rnn/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
r
encoder_1/rnn/stackPackencoder_1/rnn/strided_slice*

axis *
_output_shapes
:*
T0*
N
m
encoder_1/rnn/EqualEqualencoder_1/rnn/Shape_1encoder_1/rnn/stack*
_output_shapes
:*
T0
]
encoder_1/rnn/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
w
encoder_1/rnn/AllAllencoder_1/rnn/Equalencoder_1/rnn/Const*
_output_shapes
: *

Tidx0*
	keep_dims( 

encoder_1/rnn/Assert/ConstConst*
dtype0*
_output_shapes
: *J
valueAB? B9Expected shape for Tensor encoder_1/sequence_length:0 is 
m
encoder_1/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
_output_shapes
: *
dtype0

"encoder_1/rnn/Assert/Assert/data_0Const*
dtype0*
_output_shapes
: *J
valueAB? B9Expected shape for Tensor encoder_1/sequence_length:0 is 
s
"encoder_1/rnn/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
Ģ
encoder_1/rnn/Assert/AssertAssertencoder_1/rnn/All"encoder_1/rnn/Assert/Assert/data_0encoder_1/rnn/stack"encoder_1/rnn/Assert/Assert/data_2encoder_1/rnn/Shape_1*
	summarize*
T
2

encoder_1/rnn/CheckSeqLenIdentityencoder_1/sequence_length^encoder_1/rnn/Assert/Assert*
T0*
_output_shapes	
:
j
encoder_1/rnn/Shape_2Const*!
valueB"         *
_output_shapes
:*
dtype0
m
#encoder_1/rnn/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: 
o
%encoder_1/rnn/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
o
%encoder_1/rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
É
encoder_1/rnn/strided_slice_2StridedSliceencoder_1/rnn/Shape_2#encoder_1/rnn/strided_slice_2/stack%encoder_1/rnn/strided_slice_2/stack_1%encoder_1/rnn/strided_slice_2/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0*
end_mask *
new_axis_mask *
ellipsis_mask *

begin_mask 
m
#encoder_1/rnn/strided_slice_3/stackConst*
valueB:*
_output_shapes
:*
dtype0
o
%encoder_1/rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
o
%encoder_1/rnn/strided_slice_3/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
É
encoder_1/rnn/strided_slice_3StridedSliceencoder_1/rnn/Shape_2#encoder_1/rnn/strided_slice_3/stack%encoder_1/rnn/strided_slice_3/stack_1%encoder_1/rnn/strided_slice_3/stack_2*
end_mask *
ellipsis_mask *

begin_mask *
shrink_axis_mask*
_output_shapes
: *
new_axis_mask *
T0*
Index0
Z
encoder_1/rnn/stack_1/1Const*
dtype0*
_output_shapes
: *
value
B :

encoder_1/rnn/stack_1Packencoder_1/rnn/strided_slice_3encoder_1/rnn/stack_1/1*
_output_shapes
:*
N*

axis *
T0
^
encoder_1/rnn/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0

encoder_1/rnn/zerosFillencoder_1/rnn/stack_1encoder_1/rnn/zeros/Const*(
_output_shapes
:’’’’’’’’’*
T0
_
encoder_1/rnn/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

encoder_1/rnn/MinMinencoder_1/rnn/CheckSeqLenencoder_1/rnn/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
_
encoder_1/rnn/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 

encoder_1/rnn/MaxMaxencoder_1/rnn/CheckSeqLenencoder_1/rnn/Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
T
encoder_1/rnn/timeConst*
dtype0*
_output_shapes
: *
value	B : 
ō
encoder_1/rnn/TensorArrayTensorArrayV3encoder_1/rnn/strided_slice_2*
element_shape:*9
tensor_array_name$"encoder_1/rnn/dynamic_rnn/output_0*
dynamic_size( *
clear_after_read(*
dtype0*
_output_shapes

::
õ
encoder_1/rnn/TensorArray_1TensorArrayV3encoder_1/rnn/strided_slice_2*8
tensor_array_name#!encoder_1/rnn/dynamic_rnn/input_0*
dtype0*
element_shape:*
_output_shapes

::*
dynamic_size( *
clear_after_read(
{
&encoder_1/rnn/TensorArrayUnstack/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
~
4encoder_1/rnn/TensorArrayUnstack/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0

.encoder_1/rnn/TensorArrayUnstack/strided_sliceStridedSlice&encoder_1/rnn/TensorArrayUnstack/Shape4encoder_1/rnn/TensorArrayUnstack/strided_slice/stack6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_16encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
n
,encoder_1/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
n
,encoder_1/rnn/TensorArrayUnstack/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
ģ
&encoder_1/rnn/TensorArrayUnstack/rangeRange,encoder_1/rnn/TensorArrayUnstack/range/start.encoder_1/rnn/TensorArrayUnstack/strided_slice,encoder_1/rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:’’’’’’’’’
Ŗ
Hencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3encoder_1/rnn/TensorArray_1&encoder_1/rnn/TensorArrayUnstack/rangeencoder_1/transposeencoder_1/rnn/TensorArray_1:1*
_output_shapes
: *.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
T0
æ
encoder_1/rnn/while/EnterEnterencoder_1/rnn/time*
is_constant( *
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
Ģ
encoder_1/rnn/while/Enter_1Enterencoder_1/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
ą
encoder_1/rnn/while/Enter_2Enterencoder_1/initial_state_0_tiled*(
_output_shapes
:’’’’’’’’’*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
ą
encoder_1/rnn/while/Enter_3Enterencoder_1/initial_state_1_tiled*(
_output_shapes
:’’’’’’’’’*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
ą
encoder_1/rnn/while/Enter_4Enterencoder_1/initial_state_2_tiled*
parallel_iterations *
T0*(
_output_shapes
:’’’’’’’’’*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
ą
encoder_1/rnn/while/Enter_5Enterencoder_1/initial_state_3_tiled*
is_constant( *(
_output_shapes
:’’’’’’’’’*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 

encoder_1/rnn/while/MergeMergeencoder_1/rnn/while/Enter!encoder_1/rnn/while/NextIteration*
_output_shapes
: : *
T0*
N

encoder_1/rnn/while/Merge_1Mergeencoder_1/rnn/while/Enter_1#encoder_1/rnn/while/NextIteration_1*
N*
T0*
_output_shapes
:: 
¤
encoder_1/rnn/while/Merge_2Mergeencoder_1/rnn/while/Enter_2#encoder_1/rnn/while/NextIteration_2**
_output_shapes
:’’’’’’’’’: *
T0*
N
¤
encoder_1/rnn/while/Merge_3Mergeencoder_1/rnn/while/Enter_3#encoder_1/rnn/while/NextIteration_3*
T0*
N**
_output_shapes
:’’’’’’’’’: 
¤
encoder_1/rnn/while/Merge_4Mergeencoder_1/rnn/while/Enter_4#encoder_1/rnn/while/NextIteration_4*
T0*
N**
_output_shapes
:’’’’’’’’’: 
¤
encoder_1/rnn/while/Merge_5Mergeencoder_1/rnn/while/Enter_5#encoder_1/rnn/while/NextIteration_5*
N*
T0**
_output_shapes
:’’’’’’’’’: 
Ļ
encoder_1/rnn/while/Less/EnterEnterencoder_1/rnn/strided_slice_2*
is_constant(*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
|
encoder_1/rnn/while/LessLessencoder_1/rnn/while/Mergeencoder_1/rnn/while/Less/Enter*
_output_shapes
: *
T0
Z
encoder_1/rnn/while/LoopCondLoopCondencoder_1/rnn/while/Less*
_output_shapes
: 
®
encoder_1/rnn/while/SwitchSwitchencoder_1/rnn/while/Mergeencoder_1/rnn/while/LoopCond*,
_class"
 loc:@encoder_1/rnn/while/Merge*
_output_shapes
: : *
T0
ø
encoder_1/rnn/while/Switch_1Switchencoder_1/rnn/while/Merge_1encoder_1/rnn/while/LoopCond*
T0*
_output_shapes

::*.
_class$
" loc:@encoder_1/rnn/while/Merge_1
Ų
encoder_1/rnn/while/Switch_2Switchencoder_1/rnn/while/Merge_2encoder_1/rnn/while/LoopCond*
T0*.
_class$
" loc:@encoder_1/rnn/while/Merge_2*<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’
Ų
encoder_1/rnn/while/Switch_3Switchencoder_1/rnn/while/Merge_3encoder_1/rnn/while/LoopCond*
T0*<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*.
_class$
" loc:@encoder_1/rnn/while/Merge_3
Ų
encoder_1/rnn/while/Switch_4Switchencoder_1/rnn/while/Merge_4encoder_1/rnn/while/LoopCond*
T0*<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*.
_class$
" loc:@encoder_1/rnn/while/Merge_4
Ų
encoder_1/rnn/while/Switch_5Switchencoder_1/rnn/while/Merge_5encoder_1/rnn/while/LoopCond*
T0*.
_class$
" loc:@encoder_1/rnn/while/Merge_5*<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’
g
encoder_1/rnn/while/IdentityIdentityencoder_1/rnn/while/Switch:1*
T0*
_output_shapes
: 
m
encoder_1/rnn/while/Identity_1Identityencoder_1/rnn/while/Switch_1:1*
T0*
_output_shapes
:
}
encoder_1/rnn/while/Identity_2Identityencoder_1/rnn/while/Switch_2:1*
T0*(
_output_shapes
:’’’’’’’’’
}
encoder_1/rnn/while/Identity_3Identityencoder_1/rnn/while/Switch_3:1*(
_output_shapes
:’’’’’’’’’*
T0
}
encoder_1/rnn/while/Identity_4Identityencoder_1/rnn/while/Switch_4:1*(
_output_shapes
:’’’’’’’’’*
T0
}
encoder_1/rnn/while/Identity_5Identityencoder_1/rnn/while/Switch_5:1*
T0*(
_output_shapes
:’’’’’’’’’

+encoder_1/rnn/while/TensorArrayReadV3/EnterEnterencoder_1/rnn/TensorArray_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
T0
¹
-encoder_1/rnn/while/TensorArrayReadV3/Enter_1EnterHencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
T0

%encoder_1/rnn/while/TensorArrayReadV3TensorArrayReadV3+encoder_1/rnn/while/TensorArrayReadV3/Enterencoder_1/rnn/while/Identity-encoder_1/rnn/while/TensorArrayReadV3/Enter_1*.
_class$
" loc:@encoder_1/rnn/TensorArray_1* 
_output_shapes
:
*
dtype0
ķ
Tencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB"      *
_output_shapes
:*
dtype0
ß
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/minConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB
 *
×£½*
_output_shapes
: *
dtype0
ß
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/maxConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB
 *
×£=*
_output_shapes
: *
dtype0
Ż
\encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/RandomUniformRandomUniformTencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/shape*

seed{*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
seed2Ś*
dtype0* 
_output_shapes
:

ź
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/subSubRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/maxRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
_output_shapes
: 
ž
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/mulMul\encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/RandomUniformRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/sub* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0
š
Nencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniformAddRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/mulRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
ó
3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
VariableV2*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container 
å
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AssignAssign3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsNencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
validate_shape(* 
_output_shapes
:

¤
8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/readIdentity3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0
Ŗ
Iencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axisConst^encoder_1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
¢
Dencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concatConcatV2%encoder_1/rnn/while/TensorArrayReadV3encoder_1/rnn/while/Identity_3Iencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis* 
_output_shapes
:
*
T0*

Tidx0*
N
 
Jencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read* 
_output_shapes
:
*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
±
Dencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMulMatMulDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concatJencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
Ś
Dencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Initializer/ConstConst*
_output_shapes	
:*
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
valueB*    
ē
2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
VariableV2*
_output_shapes	
:*
dtype0*
shape:*
	container *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
shared_name 
Ó
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AssignAssign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesDencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Initializer/Const*
_output_shapes	
:*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
T0*
use_locking(

7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/readIdentity2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0

Aencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/EnterEnter7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read*
is_constant(*
_output_shapes	
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 

;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAddBiasAddDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMulAencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter*
data_formatNHWC*
T0* 
_output_shapes
:

¤
Cencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dimConst^encoder_1/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
¤
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/splitSplitCencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split

9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add/yConst^encoder_1/rnn/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
į
7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/addAdd;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add/y* 
_output_shapes
:
*
T0
Ŗ
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/SigmoidSigmoid7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add* 
_output_shapes
:
*
T0
Ę
7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mulMul;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoidencoder_1/rnn/while/Identity_2*
T0* 
_output_shapes
:

®
=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1Sigmoid9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split*
T0* 
_output_shapes
:

Ø
8encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/TanhTanh;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:1* 
_output_shapes
:
*
T0
ä
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1Mul=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_18encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
T0* 
_output_shapes
:

ß
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1Add7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1*
T0* 
_output_shapes
:

°
=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2Sigmoid;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:3* 
_output_shapes
:
*
T0
Ø
:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1Tanh9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1* 
_output_shapes
:
*
T0
ę
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2Mul=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0* 
_output_shapes
:

ķ
Tencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB"      
ß
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/minConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB
 *
×£½*
_output_shapes
: *
dtype0
ß
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB
 *
×£=
Ż
\encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/RandomUniformRandomUniformTencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/shape*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
T0*
dtype0*
seed2ū*

seed{
ź
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/subSubRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/maxRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
_output_shapes
: 
ž
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/mulMul\encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/RandomUniformRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/sub*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
T0
š
Nencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniformAddRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/mulRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:

ó
3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
VariableV2*
	container *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
shape:
*
shared_name 
å
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AssignAssign3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsNencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
¤
8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/readIdentity3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0* 
_output_shapes
:

Ŗ
Iencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axisConst^encoder_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
¶
Dencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concatConcatV29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2encoder_1/rnn/while/Identity_5Iencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
N*

Tidx0*
T0* 
_output_shapes
:

 
Jencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
±
Dencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMulMatMulDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concatJencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
Ś
Dencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Initializer/ConstConst*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
valueB*    *
dtype0*
_output_shapes	
:
ē
2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
VariableV2*
_output_shapes	
:*
dtype0*
shape:*
	container *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
shared_name 
Ó
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AssignAssign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesDencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases

7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/readIdentity2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
_output_shapes	
:

Aencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/EnterEnter7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/

;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAddBiasAddDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMulAencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC* 
_output_shapes
:

¤
Cencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dimConst^encoder_1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
¤
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/splitSplitCencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd*D
_output_shapes2
0:
:
:
:
*
	num_split*
T0

9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add/yConst^encoder_1/rnn/while/Identity*
valueB
 *  ?*
_output_shapes
: *
dtype0
į
7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/addAdd;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:29encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add/y* 
_output_shapes
:
*
T0
Ŗ
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/SigmoidSigmoid7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add* 
_output_shapes
:
*
T0
Ę
7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mulMul;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoidencoder_1/rnn/while/Identity_4* 
_output_shapes
:
*
T0
®
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1Sigmoid9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split*
T0* 
_output_shapes
:

Ø
8encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/TanhTanh;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:1* 
_output_shapes
:
*
T0
ä
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1Mul=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_18encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh* 
_output_shapes
:
*
T0
ß
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1Add7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1* 
_output_shapes
:
*
T0
°
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2Sigmoid;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:3* 
_output_shapes
:
*
T0
Ø
:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1Tanh9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1* 
_output_shapes
:
*
T0
ę
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2Mul=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
T0* 
_output_shapes
:

Ų
&encoder_1/rnn/while/GreaterEqual/EnterEnterencoder_1/rnn/CheckSeqLen*
is_constant(*
_output_shapes	
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 

 encoder_1/rnn/while/GreaterEqualGreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:
Ł
 encoder_1/rnn/while/Select/EnterEnterencoder_1/rnn/zeros*(
_output_shapes
:’’’’’’’’’*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
Ī
encoder_1/rnn/while/SelectSelect encoder_1/rnn/while/GreaterEqual encoder_1/rnn/while/Select/Enter9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2* 
_output_shapes
:
*
T0

"encoder_1/rnn/while/GreaterEqual_1GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:
Š
encoder_1/rnn/while/Select_1Select"encoder_1/rnn/while/GreaterEqual_1encoder_1/rnn/while/Identity_29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1*
T0* 
_output_shapes
:


"encoder_1/rnn/while/GreaterEqual_2GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:
Š
encoder_1/rnn/while/Select_2Select"encoder_1/rnn/while/GreaterEqual_2encoder_1/rnn/while/Identity_39encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2* 
_output_shapes
:
*
T0

"encoder_1/rnn/while/GreaterEqual_3GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
_output_shapes	
:*
T0
Š
encoder_1/rnn/while/Select_3Select"encoder_1/rnn/while/GreaterEqual_3encoder_1/rnn/while/Identity_49encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1* 
_output_shapes
:
*
T0

"encoder_1/rnn/while/GreaterEqual_4GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
_output_shapes	
:*
T0
Š
encoder_1/rnn/while/Select_4Select"encoder_1/rnn/while/GreaterEqual_4encoder_1/rnn/while/Identity_59encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0* 
_output_shapes
:


=encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterencoder_1/rnn/TensorArray*
T0*,
_class"
 loc:@encoder_1/rnn/TensorArray*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
µ
7encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3=encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterencoder_1/rnn/while/Identityencoder_1/rnn/while/Selectencoder_1/rnn/while/Identity_1*
_output_shapes
: *,
_class"
 loc:@encoder_1/rnn/TensorArray*
T0
z
encoder_1/rnn/while/add/yConst^encoder_1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
x
encoder_1/rnn/while/addAddencoder_1/rnn/while/Identityencoder_1/rnn/while/add/y*
T0*
_output_shapes
: 
l
!encoder_1/rnn/while/NextIterationNextIterationencoder_1/rnn/while/add*
_output_shapes
: *
T0

#encoder_1/rnn/while/NextIteration_1NextIteration7encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
}
#encoder_1/rnn/while/NextIteration_2NextIterationencoder_1/rnn/while/Select_1*
T0* 
_output_shapes
:

}
#encoder_1/rnn/while/NextIteration_3NextIterationencoder_1/rnn/while/Select_2* 
_output_shapes
:
*
T0
}
#encoder_1/rnn/while/NextIteration_4NextIterationencoder_1/rnn/while/Select_3* 
_output_shapes
:
*
T0
}
#encoder_1/rnn/while/NextIteration_5NextIterationencoder_1/rnn/while/Select_4* 
_output_shapes
:
*
T0
]
encoder_1/rnn/while/ExitExitencoder_1/rnn/while/Switch*
T0*
_output_shapes
: 
c
encoder_1/rnn/while/Exit_1Exitencoder_1/rnn/while/Switch_1*
T0*
_output_shapes
:
s
encoder_1/rnn/while/Exit_2Exitencoder_1/rnn/while/Switch_2*(
_output_shapes
:’’’’’’’’’*
T0
s
encoder_1/rnn/while/Exit_3Exitencoder_1/rnn/while/Switch_3*
T0*(
_output_shapes
:’’’’’’’’’
s
encoder_1/rnn/while/Exit_4Exitencoder_1/rnn/while/Switch_4*
T0*(
_output_shapes
:’’’’’’’’’
s
encoder_1/rnn/while/Exit_5Exitencoder_1/rnn/while/Switch_5*(
_output_shapes
:’’’’’’’’’*
T0
Ā
0encoder_1/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3encoder_1/rnn/TensorArrayencoder_1/rnn/while/Exit_1*,
_class"
 loc:@encoder_1/rnn/TensorArray*
_output_shapes
: 

*encoder_1/rnn/TensorArrayStack/range/startConst*
value	B : *,
_class"
 loc:@encoder_1/rnn/TensorArray*
dtype0*
_output_shapes
: 

*encoder_1/rnn/TensorArrayStack/range/deltaConst*
value	B :*,
_class"
 loc:@encoder_1/rnn/TensorArray*
_output_shapes
: *
dtype0

$encoder_1/rnn/TensorArrayStack/rangeRange*encoder_1/rnn/TensorArrayStack/range/start0encoder_1/rnn/TensorArrayStack/TensorArraySizeV3*encoder_1/rnn/TensorArrayStack/range/delta*#
_output_shapes
:’’’’’’’’’*,
_class"
 loc:@encoder_1/rnn/TensorArray*

Tidx0
§
2encoder_1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3encoder_1/rnn/TensorArray$encoder_1/rnn/TensorArrayStack/rangeencoder_1/rnn/while/Exit_1*
element_shape:
*,
_class"
 loc:@encoder_1/rnn/TensorArray*
dtype0*%
_output_shapes
:
q
encoder_1/rnn/transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          
³
encoder_1/rnn/transpose	Transpose2encoder_1/rnn/TensorArrayStack/TensorArrayGatherV3encoder_1/rnn/transpose/perm*
Tperm0*
T0*%
_output_shapes
:
Æ
6output_projection/W/Initializer/truncated_normal/shapeConst*&
_class
loc:@output_projection/W*
valueB"      *
_output_shapes
:*
dtype0
¢
5output_projection/W/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*&
_class
loc:@output_projection/W*
valueB
 *    
¤
7output_projection/W/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*&
_class
loc:@output_projection/W*
valueB
 *ĶĢĢ=

@output_projection/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6output_projection/W/Initializer/truncated_normal/shape*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0*
dtype0*
seed2æ*

seed{

4output_projection/W/Initializer/truncated_normal/mulMul@output_projection/W/Initializer/truncated_normal/TruncatedNormal7output_projection/W/Initializer/truncated_normal/stddev*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0
ö
0output_projection/W/Initializer/truncated_normalAdd4output_projection/W/Initializer/truncated_normal/mul5output_projection/W/Initializer/truncated_normal/mean*
T0*
_output_shapes
:	*&
_class
loc:@output_projection/W
±
output_projection/W
VariableV2*
	container *
dtype0*&
_class
loc:@output_projection/W*
shared_name *
_output_shapes
:	*
shape:	
ę
output_projection/W/AssignAssignoutput_projection/W0output_projection/W/Initializer/truncated_normal*
use_locking(*
T0*&
_class
loc:@output_projection/W*
validate_shape(*
_output_shapes
:	

output_projection/W/readIdentityoutput_projection/W*
T0*
_output_shapes
:	*&
_class
loc:@output_projection/W

%output_projection/b/Initializer/ConstConst*&
_class
loc:@output_projection/b*
valueB*ĶĢĢ=*
_output_shapes
:*
dtype0
§
output_projection/b
VariableV2*
shared_name *&
_class
loc:@output_projection/b*
	container *
shape:*
dtype0*
_output_shapes
:
Ö
output_projection/b/AssignAssignoutput_projection/b%output_projection/b/Initializer/Const*
use_locking(*
T0*&
_class
loc:@output_projection/b*
validate_shape(*
_output_shapes
:

output_projection/b/readIdentityoutput_projection/b*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0
ŗ
"output_projection/xw_plus_b/MatMulMatMulencoder_1/rnn/while/Exit_4output_projection/W/read*
transpose_b( *'
_output_shapes
:’’’’’’’’’*
transpose_a( *
T0
­
output_projection/xw_plus_bBiasAdd"output_projection/xw_plus_b/MatMuloutput_projection/b/read*
data_formatNHWC*
T0*'
_output_shapes
:’’’’’’’’’
s
output_projection/SoftmaxSoftmaxoutput_projection/xw_plus_b*'
_output_shapes
:’’’’’’’’’*
T0
d
"output_projection/ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0

output_projection/ArgMaxArgMaxoutput_projection/xw_plus_b"output_projection/ArgMax/dimension*#
_output_shapes
:’’’’’’’’’*
T0*

Tidx0
o

loss/ConstConst*
_output_shapes
:*
dtype0*1
value(B&"  ?¾z?`åp?bX?¤p}?  ?mē{>
j
loss/MulMuloutput_projection/xw_plus_b
loss/Const*'
_output_shapes
:’’’’’’’’’*
T0
X
	loss/CastCastcond/Merge_1*
_output_shapes
:	*

DstT0*

SrcT0
]
loss/logistic_loss/sub/yConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
h
loss/logistic_loss/subSub
loss/Constloss/logistic_loss/sub/y*
_output_shapes
:*
T0
j
loss/logistic_loss/mulMulloss/logistic_loss/sub	loss/Cast*
T0*
_output_shapes
:	
]
loss/logistic_loss/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
y
loss/logistic_loss/addAddloss/logistic_loss/add/xloss/logistic_loss/mul*
_output_shapes
:	*
T0
_
loss/logistic_loss/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
p
loss/logistic_loss/sub_1Subloss/logistic_loss/sub_1/x	loss/Cast*
T0*
_output_shapes
:	
m
loss/logistic_loss/mul_1Mulloss/logistic_loss/sub_1loss/Mul*
T0*
_output_shapes
:	
Y
loss/logistic_loss/AbsAbsloss/Mul*'
_output_shapes
:’’’’’’’’’*
T0
g
loss/logistic_loss/NegNegloss/logistic_loss/Abs*
T0*'
_output_shapes
:’’’’’’’’’
g
loss/logistic_loss/ExpExploss/logistic_loss/Neg*
T0*'
_output_shapes
:’’’’’’’’’
k
loss/logistic_loss/Log1pLog1ploss/logistic_loss/Exp*
T0*'
_output_shapes
:’’’’’’’’’
[
loss/logistic_loss/Neg_1Negloss/Mul*'
_output_shapes
:’’’’’’’’’*
T0
k
loss/logistic_loss/ReluReluloss/logistic_loss/Neg_1*
T0*'
_output_shapes
:’’’’’’’’’

loss/logistic_loss/add_1Addloss/logistic_loss/Log1ploss/logistic_loss/Relu*'
_output_shapes
:’’’’’’’’’*
T0
{
loss/logistic_loss/mul_2Mulloss/logistic_loss/addloss/logistic_loss/add_1*
T0*
_output_shapes
:	
w
loss/logistic_lossAddloss/logistic_loss/mul_1loss/logistic_loss/mul_2*
T0*
_output_shapes
:	
]
loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
o
loss/SumSumloss/logistic_lossloss/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
]
loss/Const_2Const*
dtype0*
_output_shapes
:*
valueB"       
q
	loss/MeanMeanloss/logistic_lossloss/Const_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
[
accuracy/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
t
accuracy/ArgMaxArgMaxcond/Merge_1accuracy/ArgMax/dimension*
_output_shapes	
:*
T0*

Tidx0
h
accuracy/EqualEqualoutput_projection/ArgMaxaccuracy/ArgMax*
T0	*
_output_shapes	
:
Z
accuracy/CastCastaccuracy/Equal*

SrcT0
*
_output_shapes	
:*

DstT0
X
accuracy/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
v
accuracy/accuracyMeanaccuracy/Castaccuracy/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
`
learning_rate/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *·Q8
Y
learning_rate/CastCastVariable/read*

SrcT0*
_output_shapes
: *

DstT0
Y
learning_rate/Cast_1/xConst*
_output_shapes
: *
dtype0*
value
B :'
d
learning_rate/Cast_1Castlearning_rate/Cast_1/x*
_output_shapes
: *

DstT0*

SrcT0
[
learning_rate/Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *Āu?
k
learning_rate/truedivRealDivlearning_rate/Castlearning_rate/Cast_1*
T0*
_output_shapes
: 
T
learning_rate/FloorFloorlearning_rate/truediv*
_output_shapes
: *
T0
f
learning_rate/PowPowlearning_rate/Cast_2/xlearning_rate/Floor*
_output_shapes
: *
T0
e
learning_rateMullearning_rate/learning_ratelearning_rate/Pow*
T0*
_output_shapes
: 
`
gradients/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      
T
gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
b
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
:	*
T0
S
gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
ø
gradients/f_count_1Entergradients/f_count*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
T0*
N*
_output_shapes
: : 
l
gradients/SwitchSwitchgradients/Mergeencoder_1/rnn/while/LoopCond*
T0*
_output_shapes
: : 
p
gradients/Add/yConst^encoder_1/rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
T0*
_output_shapes
: 
¹
gradients/NextIterationNextIterationgradients/AddA^gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPushA^gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPushA^gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPushA^gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPushZ^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPushb^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPushe^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPushZ^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPushb^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPushe^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPushc^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPush*
_output_shapes
: *
T0
N
gradients/f_count_2Exitgradients/Switch*
_output_shapes
: *
T0
S
gradients/b_countConst*
dtype0*
_output_shapes
: *
value	B :
Ä
gradients/b_count_1Entergradients/f_count_2*
parallel_iterations *
T0*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
N*
T0*
_output_shapes
: : 
Ė
gradients/GreaterEqual/EnterEntergradients/b_count*
parallel_iterations *
T0*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(
x
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
O
gradients/b_count_2LoopCondgradients/GreaterEqual*
_output_shapes
: 
g
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
T0*
_output_shapes
: : 
i
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 

gradients/NextIteration_1NextIterationgradients/Sub>^gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/b_sync*
_output_shapes
: *
T0
P
gradients/b_count_3Exitgradients/Switch_1*
T0*
_output_shapes
: 
x
'gradients/loss/logistic_loss_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
z
)gradients/loss/logistic_loss_grad/Shape_1Const*
valueB"      *
_output_shapes
:*
dtype0
į
7gradients/loss/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/loss/logistic_loss_grad/Shape)gradients/loss/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
µ
%gradients/loss/logistic_loss_grad/SumSumgradients/Fill7gradients/loss/logistic_loss_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¼
)gradients/loss/logistic_loss_grad/ReshapeReshape%gradients/loss/logistic_loss_grad/Sum'gradients/loss/logistic_loss_grad/Shape*
T0*
_output_shapes
:	*
Tshape0
¹
'gradients/loss/logistic_loss_grad/Sum_1Sumgradients/Fill9gradients/loss/logistic_loss_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ā
+gradients/loss/logistic_loss_grad/Reshape_1Reshape'gradients/loss/logistic_loss_grad/Sum_1)gradients/loss/logistic_loss_grad/Shape_1*
Tshape0*
_output_shapes
:	*
T0
~
-gradients/loss/logistic_loss/mul_1_grad/ShapeConst*
valueB"      *
_output_shapes
:*
dtype0
w
/gradients/loss/logistic_loss/mul_1_grad/Shape_1Shapeloss/Mul*
T0*
out_type0*
_output_shapes
:
ó
=gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/mul_1_grad/Shape/gradients/loss/logistic_loss/mul_1_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0

+gradients/loss/logistic_loss/mul_1_grad/mulMul)gradients/loss/logistic_loss_grad/Reshapeloss/Mul*
_output_shapes
:	*
T0
Ž
+gradients/loss/logistic_loss/mul_1_grad/SumSum+gradients/loss/logistic_loss/mul_1_grad/mul=gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ī
/gradients/loss/logistic_loss/mul_1_grad/ReshapeReshape+gradients/loss/logistic_loss/mul_1_grad/Sum-gradients/loss/logistic_loss/mul_1_grad/Shape*
T0*
_output_shapes
:	*
Tshape0
£
-gradients/loss/logistic_loss/mul_1_grad/mul_1Mulloss/logistic_loss/sub_1)gradients/loss/logistic_loss_grad/Reshape*
T0*
_output_shapes
:	
ä
-gradients/loss/logistic_loss/mul_1_grad/Sum_1Sum-gradients/loss/logistic_loss/mul_1_grad/mul_1?gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ü
1gradients/loss/logistic_loss/mul_1_grad/Reshape_1Reshape-gradients/loss/logistic_loss/mul_1_grad/Sum_1/gradients/loss/logistic_loss/mul_1_grad/Shape_1*'
_output_shapes
:’’’’’’’’’*
Tshape0*
T0
~
-gradients/loss/logistic_loss/mul_2_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      

/gradients/loss/logistic_loss/mul_2_grad/Shape_1Shapeloss/logistic_loss/add_1*
_output_shapes
:*
out_type0*
T0
ó
=gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/mul_2_grad/Shape/gradients/loss/logistic_loss/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
£
+gradients/loss/logistic_loss/mul_2_grad/mulMul+gradients/loss/logistic_loss_grad/Reshape_1loss/logistic_loss/add_1*
_output_shapes
:	*
T0
Ž
+gradients/loss/logistic_loss/mul_2_grad/SumSum+gradients/loss/logistic_loss/mul_2_grad/mul=gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ī
/gradients/loss/logistic_loss/mul_2_grad/ReshapeReshape+gradients/loss/logistic_loss/mul_2_grad/Sum-gradients/loss/logistic_loss/mul_2_grad/Shape*
T0*
Tshape0*
_output_shapes
:	
£
-gradients/loss/logistic_loss/mul_2_grad/mul_1Mulloss/logistic_loss/add+gradients/loss/logistic_loss_grad/Reshape_1*
_output_shapes
:	*
T0
ä
-gradients/loss/logistic_loss/mul_2_grad/Sum_1Sum-gradients/loss/logistic_loss/mul_2_grad/mul_1?gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ü
1gradients/loss/logistic_loss/mul_2_grad/Reshape_1Reshape-gradients/loss/logistic_loss/mul_2_grad/Sum_1/gradients/loss/logistic_loss/mul_2_grad/Shape_1*
Tshape0*'
_output_shapes
:’’’’’’’’’*
T0

-gradients/loss/logistic_loss/add_1_grad/ShapeShapeloss/logistic_loss/Log1p*
_output_shapes
:*
out_type0*
T0

/gradients/loss/logistic_loss/add_1_grad/Shape_1Shapeloss/logistic_loss/Relu*
out_type0*
_output_shapes
:*
T0
ó
=gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/add_1_grad/Shape/gradients/loss/logistic_loss/add_1_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
ä
+gradients/loss/logistic_loss/add_1_grad/SumSum1gradients/loss/logistic_loss/mul_2_grad/Reshape_1=gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ö
/gradients/loss/logistic_loss/add_1_grad/ReshapeReshape+gradients/loss/logistic_loss/add_1_grad/Sum-gradients/loss/logistic_loss/add_1_grad/Shape*'
_output_shapes
:’’’’’’’’’*
Tshape0*
T0
č
-gradients/loss/logistic_loss/add_1_grad/Sum_1Sum1gradients/loss/logistic_loss/mul_2_grad/Reshape_1?gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ü
1gradients/loss/logistic_loss/add_1_grad/Reshape_1Reshape-gradients/loss/logistic_loss/add_1_grad/Sum_1/gradients/loss/logistic_loss/add_1_grad/Shape_1*'
_output_shapes
:’’’’’’’’’*
Tshape0*
T0
¤
-gradients/loss/logistic_loss/Log1p_grad/add/xConst0^gradients/loss/logistic_loss/add_1_grad/Reshape*
_output_shapes
: *
dtype0*
valueB
 *  ?
«
+gradients/loss/logistic_loss/Log1p_grad/addAdd-gradients/loss/logistic_loss/Log1p_grad/add/xloss/logistic_loss/Exp*'
_output_shapes
:’’’’’’’’’*
T0

2gradients/loss/logistic_loss/Log1p_grad/Reciprocal
Reciprocal+gradients/loss/logistic_loss/Log1p_grad/add*
T0*'
_output_shapes
:’’’’’’’’’
É
+gradients/loss/logistic_loss/Log1p_grad/mulMul/gradients/loss/logistic_loss/add_1_grad/Reshape2gradients/loss/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:’’’’’’’’’*
T0
¹
/gradients/loss/logistic_loss/Relu_grad/ReluGradReluGrad1gradients/loss/logistic_loss/add_1_grad/Reshape_1loss/logistic_loss/Relu*
T0*'
_output_shapes
:’’’’’’’’’
§
)gradients/loss/logistic_loss/Exp_grad/mulMul+gradients/loss/logistic_loss/Log1p_grad/mulloss/logistic_loss/Exp*
T0*'
_output_shapes
:’’’’’’’’’

+gradients/loss/logistic_loss/Neg_1_grad/NegNeg/gradients/loss/logistic_loss/Relu_grad/ReluGrad*'
_output_shapes
:’’’’’’’’’*
T0

)gradients/loss/logistic_loss/Neg_grad/NegNeg)gradients/loss/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:’’’’’’’’’
n
*gradients/loss/logistic_loss/Abs_grad/SignSignloss/Mul*'
_output_shapes
:’’’’’’’’’*
T0
¹
)gradients/loss/logistic_loss/Abs_grad/mulMul)gradients/loss/logistic_loss/Neg_grad/Neg*gradients/loss/logistic_loss/Abs_grad/Sign*'
_output_shapes
:’’’’’’’’’*
T0
¢
gradients/AddNAddN1gradients/loss/logistic_loss/mul_1_grad/Reshape_1+gradients/loss/logistic_loss/Neg_1_grad/Neg)gradients/loss/logistic_loss/Abs_grad/mul*D
_class:
86loc:@gradients/loss/logistic_loss/mul_1_grad/Reshape_1*'
_output_shapes
:’’’’’’’’’*
T0*
N
x
gradients/loss/Mul_grad/ShapeShapeoutput_projection/xw_plus_b*
_output_shapes
:*
out_type0*
T0
i
gradients/loss/Mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ć
-gradients/loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/Mul_grad/Shapegradients/loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
p
gradients/loss/Mul_grad/mulMulgradients/AddN
loss/Const*
T0*'
_output_shapes
:’’’’’’’’’
®
gradients/loss/Mul_grad/SumSumgradients/loss/Mul_grad/mul-gradients/loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¦
gradients/loss/Mul_grad/ReshapeReshapegradients/loss/Mul_grad/Sumgradients/loss/Mul_grad/Shape*
T0*'
_output_shapes
:’’’’’’’’’*
Tshape0

gradients/loss/Mul_grad/mul_1Muloutput_projection/xw_plus_bgradients/AddN*'
_output_shapes
:’’’’’’’’’*
T0
“
gradients/loss/Mul_grad/Sum_1Sumgradients/loss/Mul_grad/mul_1/gradients/loss/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

!gradients/loss/Mul_grad/Reshape_1Reshapegradients/loss/Mul_grad/Sum_1gradients/loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
¢
6gradients/output_projection/xw_plus_b_grad/BiasAddGradBiasAddGradgradients/loss/Mul_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
Ö
8gradients/output_projection/xw_plus_b/MatMul_grad/MatMulMatMulgradients/loss/Mul_grad/Reshapeoutput_projection/W/read*
transpose_b(*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
T0
Ń
:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1MatMulencoder_1/rnn/while/Exit_4gradients/loss/Mul_grad/Reshape*
transpose_b( *
T0*
_output_shapes
:	*
transpose_a(
`
gradients/zeros_like	ZerosLikeencoder_1/rnn/while/Exit_1*
_output_shapes
:*
T0
r
gradients/zeros_like_1	ZerosLikeencoder_1/rnn/while/Exit_2*
T0*(
_output_shapes
:’’’’’’’’’
r
gradients/zeros_like_2	ZerosLikeencoder_1/rnn/while/Exit_3*
T0*(
_output_shapes
:’’’’’’’’’
r
gradients/zeros_like_3	ZerosLikeencoder_1/rnn/while/Exit_5*(
_output_shapes
:’’’’’’’’’*
T0

0gradients/encoder_1/rnn/while/Exit_4_grad/b_exitEnter8gradients/output_projection/xw_plus_b/MatMul_grad/MatMul*(
_output_shapes
:’’’’’’’’’*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
ä
0gradients/encoder_1/rnn/while/Exit_1_grad/b_exitEntergradients/zeros_like*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
ö
0gradients/encoder_1/rnn/while/Exit_2_grad/b_exitEntergradients/zeros_like_1*
parallel_iterations *
T0*(
_output_shapes
:’’’’’’’’’*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
ö
0gradients/encoder_1/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_like_2*
is_constant( *(
_output_shapes
:’’’’’’’’’*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
ö
0gradients/encoder_1/rnn/while/Exit_5_grad/b_exitEntergradients/zeros_like_3*(
_output_shapes
:’’’’’’’’’*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
ź
4gradients/encoder_1/rnn/while/Switch_4_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_4_grad/b_exit;gradients/encoder_1/rnn/while/Switch_4_grad_1/NextIteration*
N*
T0**
_output_shapes
:’’’’’’’’’: 
ź
4gradients/encoder_1/rnn/while/Switch_2_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_2_grad/b_exit;gradients/encoder_1/rnn/while/Switch_2_grad_1/NextIteration**
_output_shapes
:’’’’’’’’’: *
T0*
N
ź
4gradients/encoder_1/rnn/while/Switch_3_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_3_grad/b_exit;gradients/encoder_1/rnn/while/Switch_3_grad_1/NextIteration**
_output_shapes
:’’’’’’’’’: *
T0*
N
ź
4gradients/encoder_1/rnn/while/Switch_5_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_5_grad/b_exit;gradients/encoder_1/rnn/while/Switch_5_grad_1/NextIteration*
T0*
N**
_output_shapes
:’’’’’’’’’: 

1gradients/encoder_1/rnn/while/Merge_4_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_4_grad/b_switch*4
_output_shapes"
 :’’’’’’’’’:
*
T0

1gradients/encoder_1/rnn/while/Merge_2_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*
T0*4
_output_shapes"
 :’’’’’’’’’:
*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_2_grad/b_switch

1gradients/encoder_1/rnn/while/Merge_3_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*4
_output_shapes"
 :’’’’’’’’’:
*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_3_grad/b_switch*
T0

1gradients/encoder_1/rnn/while/Merge_5_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_5_grad/b_switchgradients/b_count_2*4
_output_shapes"
 :’’’’’’’’’:
*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_5_grad/b_switch*
T0

/gradients/encoder_1/rnn/while/Enter_4_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_4_grad/Switch*(
_output_shapes
:’’’’’’’’’*
T0

/gradients/encoder_1/rnn/while/Enter_2_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_2_grad/Switch*
T0*(
_output_shapes
:’’’’’’’’’

/gradients/encoder_1/rnn/while/Enter_3_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_3_grad/Switch*(
_output_shapes
:’’’’’’’’’*
T0

/gradients/encoder_1/rnn/while/Enter_5_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_5_grad/Switch*
T0*(
_output_shapes
:’’’’’’’’’

4gradients/encoder_1/initial_state_2_tiled_grad/ShapeConst*
valueB"      *
_output_shapes
:*
dtype0
Ū
4gradients/encoder_1/initial_state_2_tiled_grad/stackPack)encoder_1/initial_state_2_tiled/multiples4gradients/encoder_1/initial_state_2_tiled_grad/Shape*
N*
T0*
_output_shapes

:*

axis 

=gradients/encoder_1/initial_state_2_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_2_tiled_grad/stack*
_output_shapes
: *
T0

>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
ć
<gradients/encoder_1/initial_state_2_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_2_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub/y*
T0*
_output_shapes
: 

Dgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 

Dgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ŗ
>gradients/encoder_1/initial_state_2_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_2_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/delta*

Tidx0*
_output_shapes
:
č
>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_2_tiled_grad/transpose/Range*
T0*
_output_shapes
:
ń
8gradients/encoder_1/initial_state_2_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_2_tiled_grad/stack>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub_1*
Tperm0*
_output_shapes

:*
T0

<gradients/encoder_1/initial_state_2_tiled_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
’’’’’’’’’
ģ
6gradients/encoder_1/initial_state_2_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_2_tiled_grad/transpose<gradients/encoder_1/initial_state_2_tiled_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
u
3gradients/encoder_1/initial_state_2_tiled_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
|
:gradients/encoder_1/initial_state_2_tiled_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
|
:gradients/encoder_1/initial_state_2_tiled_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :

4gradients/encoder_1/initial_state_2_tiled_grad/rangeRange:gradients/encoder_1/initial_state_2_tiled_grad/range/start3gradients/encoder_1/initial_state_2_tiled_grad/Size:gradients/encoder_1/initial_state_2_tiled_grad/range/delta*
_output_shapes
:*

Tidx0

8gradients/encoder_1/initial_state_2_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_4_grad/Exit6gradients/encoder_1/initial_state_2_tiled_grad/Reshape*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
Tshape0*
T0
š
2gradients/encoder_1/initial_state_2_tiled_grad/SumSum8gradients/encoder_1/initial_state_2_tiled_grad/Reshape_14gradients/encoder_1/initial_state_2_tiled_grad/range*
_output_shapes
:	*
T0*
	keep_dims( *

Tidx0
·
<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4
É
?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc*
is_constant(*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
§
@gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_4^gradients/Add*
_output_shapes
:*
swap_memory( *1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
T0
Ü
Hgradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 

?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*(
_output_shapes
:’’’’’’’’’*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4

=gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/b_syncControlTrigger@^gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopa^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopd^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopa^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopd^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopb^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop
·
6gradients/encoder_1/rnn/while/Select_3_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop*(
_output_shapes
:’’’’’’’’’*
T0
·
8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*

stack_name *
_output_shapes
:
Å
;gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
T0*
is_constant(
§
<gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_3^gradients/Add*
_output_shapes
:*
swap_memory( *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
T0

Ų
Dgradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc*
T0*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/

;gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
_output_shapes	
:

2gradients/encoder_1/rnn/while/Select_3_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_4_grad/Switch:16gradients/encoder_1/rnn/while/Select_3_grad/zeros_like*
T0* 
_output_shapes
:


4gradients/encoder_1/rnn/while/Select_3_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_3_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_4_grad/Switch:1*
T0* 
_output_shapes
:


4gradients/encoder_1/initial_state_0_tiled_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      
Ū
4gradients/encoder_1/initial_state_0_tiled_grad/stackPack)encoder_1/initial_state_0_tiled/multiples4gradients/encoder_1/initial_state_0_tiled_grad/Shape*

axis *
_output_shapes

:*
T0*
N

=gradients/encoder_1/initial_state_0_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_0_tiled_grad/stack*
_output_shapes
: *
T0

>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
ć
<gradients/encoder_1/initial_state_0_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_0_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub/y*
T0*
_output_shapes
: 

Dgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/startConst*
_output_shapes
: *
dtype0*
value	B : 

Dgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
ŗ
>gradients/encoder_1/initial_state_0_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_0_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/delta*

Tidx0*
_output_shapes
:
č
>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_0_tiled_grad/transpose/Range*
T0*
_output_shapes
:
ń
8gradients/encoder_1/initial_state_0_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_0_tiled_grad/stack>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub_1*
Tperm0*
_output_shapes

:*
T0

<gradients/encoder_1/initial_state_0_tiled_grad/Reshape/shapeConst*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:
ģ
6gradients/encoder_1/initial_state_0_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_0_tiled_grad/transpose<gradients/encoder_1/initial_state_0_tiled_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
u
3gradients/encoder_1/initial_state_0_tiled_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
|
:gradients/encoder_1/initial_state_0_tiled_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
|
:gradients/encoder_1/initial_state_0_tiled_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :

4gradients/encoder_1/initial_state_0_tiled_grad/rangeRange:gradients/encoder_1/initial_state_0_tiled_grad/range/start3gradients/encoder_1/initial_state_0_tiled_grad/Size:gradients/encoder_1/initial_state_0_tiled_grad/range/delta*
_output_shapes
:*

Tidx0

8gradients/encoder_1/initial_state_0_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_2_grad/Exit6gradients/encoder_1/initial_state_0_tiled_grad/Reshape*
T0*
Tshape0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
š
2gradients/encoder_1/initial_state_0_tiled_grad/SumSum8gradients/encoder_1/initial_state_0_tiled_grad/Reshape_14gradients/encoder_1/initial_state_0_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	
·
<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_accStack*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*

stack_name *
_output_shapes
:
É
?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2
§
@gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_2^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *1
_class'
%#loc:@encoder_1/rnn/while/Identity_2
Ü
Hgradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*
T0*
is_constant(

?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*(
_output_shapes
:’’’’’’’’’*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2
·
6gradients/encoder_1/rnn/while/Select_1_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop*(
_output_shapes
:’’’’’’’’’*
T0
·
8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*

stack_name *
_output_shapes
:
Å
;gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
T0*
is_constant(
§
<gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_1^gradients/Add*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
_output_shapes
:*
swap_memory( *
T0

Ų
Dgradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc*
T0*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/

;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
_output_shapes	
:

2gradients/encoder_1/rnn/while/Select_1_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_2_grad/Switch:16gradients/encoder_1/rnn/while/Select_1_grad/zeros_like* 
_output_shapes
:
*
T0

4gradients/encoder_1/rnn/while/Select_1_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_1_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_2_grad/Switch:1* 
_output_shapes
:
*
T0

4gradients/encoder_1/initial_state_1_tiled_grad/ShapeConst*
valueB"      *
_output_shapes
:*
dtype0
Ū
4gradients/encoder_1/initial_state_1_tiled_grad/stackPack)encoder_1/initial_state_1_tiled/multiples4gradients/encoder_1/initial_state_1_tiled_grad/Shape*
N*
T0*
_output_shapes

:*

axis 

=gradients/encoder_1/initial_state_1_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_1_tiled_grad/stack*
_output_shapes
: *
T0

>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub/yConst*
_output_shapes
: *
dtype0*
value	B :
ć
<gradients/encoder_1/initial_state_1_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_1_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub/y*
_output_shapes
: *
T0

Dgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/startConst*
_output_shapes
: *
dtype0*
value	B : 

Dgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
ŗ
>gradients/encoder_1/initial_state_1_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_1_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/delta*

Tidx0*
_output_shapes
:
č
>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_1_tiled_grad/transpose/Range*
T0*
_output_shapes
:
ń
8gradients/encoder_1/initial_state_1_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_1_tiled_grad/stack>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub_1*
Tperm0*
T0*
_output_shapes

:

<gradients/encoder_1/initial_state_1_tiled_grad/Reshape/shapeConst*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:
ģ
6gradients/encoder_1/initial_state_1_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_1_tiled_grad/transpose<gradients/encoder_1/initial_state_1_tiled_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
u
3gradients/encoder_1/initial_state_1_tiled_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
|
:gradients/encoder_1/initial_state_1_tiled_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
|
:gradients/encoder_1/initial_state_1_tiled_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :

4gradients/encoder_1/initial_state_1_tiled_grad/rangeRange:gradients/encoder_1/initial_state_1_tiled_grad/range/start3gradients/encoder_1/initial_state_1_tiled_grad/Size:gradients/encoder_1/initial_state_1_tiled_grad/range/delta*
_output_shapes
:*

Tidx0

8gradients/encoder_1/initial_state_1_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_3_grad/Exit6gradients/encoder_1/initial_state_1_tiled_grad/Reshape*
Tshape0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
T0
š
2gradients/encoder_1/initial_state_1_tiled_grad/SumSum8gradients/encoder_1/initial_state_1_tiled_grad/Reshape_14gradients/encoder_1/initial_state_1_tiled_grad/range*
_output_shapes
:	*
T0*
	keep_dims( *

Tidx0
·
<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_accStack*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*

stack_name *
_output_shapes
:
É
?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc*
is_constant(*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
§
@gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_3^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *1
_class'
%#loc:@encoder_1/rnn/while/Identity_3
Ü
Hgradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*(
_output_shapes
:’’’’’’’’’
·
6gradients/encoder_1/rnn/while/Select_2_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop*(
_output_shapes
:’’’’’’’’’*
T0
·
8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
_output_shapes
:*

stack_name 
Å
;gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2
§
<gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_2^gradients/Add*
T0
*
_output_shapes
:*
swap_memory( *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2
Ų
Dgradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*
_output_shapes	
:*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2

2gradients/encoder_1/rnn/while/Select_2_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_3_grad/Switch:16gradients/encoder_1/rnn/while/Select_2_grad/zeros_like*
T0* 
_output_shapes
:


4gradients/encoder_1/rnn/while/Select_2_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_2_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_3_grad/Switch:1* 
_output_shapes
:
*
T0

4gradients/encoder_1/initial_state_3_tiled_grad/ShapeConst*
valueB"      *
_output_shapes
:*
dtype0
Ū
4gradients/encoder_1/initial_state_3_tiled_grad/stackPack)encoder_1/initial_state_3_tiled/multiples4gradients/encoder_1/initial_state_3_tiled_grad/Shape*
N*
T0*
_output_shapes

:*

axis 

=gradients/encoder_1/initial_state_3_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_3_tiled_grad/stack*
_output_shapes
: *
T0

>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
ć
<gradients/encoder_1/initial_state_3_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_3_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub/y*
T0*
_output_shapes
: 

Dgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 

Dgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ŗ
>gradients/encoder_1/initial_state_3_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_3_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/delta*

Tidx0*
_output_shapes
:
č
>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_3_tiled_grad/transpose/Range*
_output_shapes
:*
T0
ń
8gradients/encoder_1/initial_state_3_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_3_tiled_grad/stack>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub_1*
Tperm0*
_output_shapes

:*
T0

<gradients/encoder_1/initial_state_3_tiled_grad/Reshape/shapeConst*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:
ģ
6gradients/encoder_1/initial_state_3_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_3_tiled_grad/transpose<gradients/encoder_1/initial_state_3_tiled_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
u
3gradients/encoder_1/initial_state_3_tiled_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
|
:gradients/encoder_1/initial_state_3_tiled_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
|
:gradients/encoder_1/initial_state_3_tiled_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :

4gradients/encoder_1/initial_state_3_tiled_grad/rangeRange:gradients/encoder_1/initial_state_3_tiled_grad/range/start3gradients/encoder_1/initial_state_3_tiled_grad/Size:gradients/encoder_1/initial_state_3_tiled_grad/range/delta*
_output_shapes
:*

Tidx0

8gradients/encoder_1/initial_state_3_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_5_grad/Exit6gradients/encoder_1/initial_state_3_tiled_grad/Reshape*
Tshape0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
T0
š
2gradients/encoder_1/initial_state_3_tiled_grad/SumSum8gradients/encoder_1/initial_state_3_tiled_grad/Reshape_14gradients/encoder_1/initial_state_3_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	
·
<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_accStack*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*

stack_name *
_output_shapes
:
É
?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
§
@gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_5^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *1
_class'
%#loc:@encoder_1/rnn/while/Identity_5
Ü
Hgradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*(
_output_shapes
:’’’’’’’’’*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5
·
6gradients/encoder_1/rnn/while/Select_4_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop*(
_output_shapes
:’’’’’’’’’*
T0
·
8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_accStack*
	elem_type0
*

stack_name *
_output_shapes
:*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4
Å
;gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
T0
§
<gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_4^gradients/Add*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
_output_shapes
:*
swap_memory( *
T0

Ų
Dgradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
T0

;gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
_output_shapes	
:

2gradients/encoder_1/rnn/while/Select_4_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_5_grad/Switch:16gradients/encoder_1/rnn/while/Select_4_grad/zeros_like* 
_output_shapes
:
*
T0

4gradients/encoder_1/rnn/while/Select_4_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_4_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_5_grad/Switch:1* 
_output_shapes
:
*
T0
Æ
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
±
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
é
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
T0*
is_constant(

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnter:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1^gradients/Add*
T0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
_output_shapes
:*
swap_memory( 
¤
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
T0*
is_constant(
Ó
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1

Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mulMul4gradients/encoder_1/rnn/while/Select_4_grad/Select_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop*
T0* 
_output_shapes
:

Į
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape*
T0* 
_output_shapes
:
*
Tshape0
ī
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
T0

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2
«
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
T0
Ś
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2

Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop4gradients/encoder_1/rnn/while/Select_4_grad/Select_1* 
_output_shapes
:
*
T0
Ē
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ø
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1* 
_output_shapes
:
*
Tshape0*
T0
½
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape*
T0* 
_output_shapes
:

“
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1*
T0* 
_output_shapes
:


gradients/AddN_1AddN4gradients/encoder_1/rnn/while/Select_3_grad/Select_1Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGrad*
T0*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_3_grad/Select_1*
N* 
_output_shapes
:

Æ
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
±
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/SumSumgradients/AddN_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0

Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ø
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1* 
_output_shapes
:
*
Tshape0*
T0
­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ShapeConst^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
¬
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1Shapeencoder_1/rnn/while/Identity_4*
out_type0*
_output_shapes
:*
T0

bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStack*
	elem_type0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:
Å
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1
Ó
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1^gradients/Add*
_output_shapes
:*
swap_memory( *a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
T0
Ų
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
T0*
is_constant(

egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*
	elem_type0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
_output_shapes
:
ē
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shapeegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mulMulPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop*
T0* 
_output_shapes
:

»
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumSumJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¬
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape* 
_output_shapes
:
*
Tshape0*
T0
ź
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_accStack*
	elem_type0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
_output_shapes
:*

stack_name 

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnter;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid^gradients/Add*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
_output_shapes
:*
swap_memory( *
T0
„
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
Ō
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid
§
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1MulUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape* 
_output_shapes
:
*
T0
Į
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ń
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*(
_output_shapes
:’’’’’’’’’*
Tshape0
Æ
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
±
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
ē
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_accStack*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
_output_shapes
:*

stack_name 

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0*
is_constant(

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnter8encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh^gradients/Add*
_output_shapes
:*
swap_memory( *K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0
¢
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc*
T0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
Ń
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh* 
_output_shapes
:

©
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mulMulRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop*
T0* 
_output_shapes
:

Į
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape* 
_output_shapes
:
*
Tshape0*
T0
ī
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1
«
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
Ś
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1
­
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1*
T0* 
_output_shapes
:

Ē
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ø
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1* 
_output_shapes
:
*
Tshape0*
T0
·
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape*
T0* 
_output_shapes
:


gradients/AddN_2AddN2gradients/encoder_1/rnn/while/Select_3_grad/SelectPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_3_grad/Select* 
_output_shapes
:
*
T0*
N
½
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape* 
_output_shapes
:
*
T0
²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1* 
_output_shapes
:
*
T0
­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
”
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
Š
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
Ē
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumSumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¬
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape*
T0*
Tshape0* 
_output_shapes
:

Ė
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1SumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ø
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0

;gradients/encoder_1/rnn/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_2*
T0* 
_output_shapes
:

õ
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim
 
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc*
is_constant(*
T0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
£
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPush	StackPushXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnterCencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim^gradients/Add*
_output_shapes
:*
swap_memory( *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
T0
³
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
Ų
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPopStackPopagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop/RefEnter^gradients/Sub*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
_output_shapes
: 
Ė
Ogradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concatConcatV2Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop* 
_output_shapes
:
*
N*
T0*

Tidx0
ó
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
data_formatNHWC*
T0*
_output_shapes	
:
Ą
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read* 
_output_shapes
:
*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
č
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMulMatMulOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
T0* 
_output_shapes
:
*
transpose_a( 

bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat
»
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat
æ
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat^gradients/Add*
_output_shapes
:*
swap_memory( *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
T0
Ī
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
T0*
is_constant(
ż
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat* 
_output_shapes
:

ļ
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1MatMulegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
transpose_b( * 
_output_shapes
:
*
transpose_a(*
T0
„
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_accConst*
_output_shapes	
:*
dtype0*
valueB*    
Ń
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes	
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
Ģ
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:: 
ż
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
::
“
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/AddAddYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch:1Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
ė
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:*
T0
ß
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:
Ŗ
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_accStack*
	elem_type0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*

stack_name 
¶
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
æ
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPush	StackPush`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnterIencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis^gradients/Add*
T0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*
swap_memory( 
É
igradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
T0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
ī
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopStackPopigradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnter^gradients/Sub*
	elem_type0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
_output_shapes
: 
Ą
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/modFloorMod`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
ŗ
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
¹
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Shape_1Shapeencoder_1/rnn/while/Identity_5*
T0*
_output_shapes
:*
out_type0
ö
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2
¬
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2
„
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPush	StackPushcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2^gradients/Add*
T0*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:*
swap_memory( 
æ
lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2
ī
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopStackPoplgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnter^gradients/Sub*
	elem_type0*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2* 
_output_shapes
:

Ī
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeNShapeNcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop* 
_output_shapes
::*
N*
out_type0*
T0
®
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffsetConcatOffsetWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/modZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
“
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/SliceSliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffsetZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN*
Index0*
T0* 
_output_shapes
:

Ā
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice_1SliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMulbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffset:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:’’’’’’’’’
ø
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_accConst*
dtype0* 
_output_shapes
:
*
valueB
*    
č
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1Enter_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations * 
_output_shapes
:
*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
ģ
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2Mergeagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIteration*"
_output_shapes
:
: *
N*
T0

`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/SwitchSwitchagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_output_shapes
:
:
*
T0
Ń
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/AddAddbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0

ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIterationNextIteration]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:

ö
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3Exit`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
*
T0
¦
gradients/AddN_3AddN4gradients/encoder_1/rnn/while/Select_2_grad/Select_1Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_2_grad/Select_1* 
_output_shapes
:
*
T0*
N
Æ
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
±
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
é
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnter:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1
¤
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1
Ó
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1
ē
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mulMulgradients/AddN_3Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop*
T0* 
_output_shapes
:

Į
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0
ī
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2^gradients/Add*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
_output_shapes
:*
swap_memory( 
«
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
Ś
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2* 
_output_shapes
:

ė
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopgradients/AddN_3* 
_output_shapes
:
*
T0
Ē
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ø
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*
Tshape0* 
_output_shapes
:
*
T0
¤
gradients/AddN_4AddN2gradients/encoder_1/rnn/while/Select_4_grad/Select[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice_1*
T0*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_4_grad/Select*
N* 
_output_shapes
:

½
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape*
T0* 
_output_shapes
:

“
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1* 
_output_shapes
:
*
T0

;gradients/encoder_1/rnn/while/Switch_5_grad_1/NextIterationNextIterationgradients/AddN_4* 
_output_shapes
:
*
T0

gradients/AddN_5AddN4gradients/encoder_1/rnn/while/Select_1_grad/Select_1Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGrad* 
_output_shapes
:
*
N*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_1_grad/Select_1*
T0
Æ
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
±
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/SumSumgradients/AddN_5^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0

Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_5`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ø
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*
T0*
Tshape0* 
_output_shapes
:

­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
¬
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1Shapeencoder_1/rnn/while/Identity_2*
T0*
_output_shapes
:*
out_type0

bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStack*
	elem_type0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*

stack_name 
Å
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1
Ó
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1^gradients/Add*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*
swap_memory( *
T0
Ų
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
T0*
is_constant(

egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
:*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1
ē
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shapeegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0

Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mulMulPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop* 
_output_shapes
:
*
T0
»
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumSumJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¬
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0
ź
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_accStack*
	elem_type0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*

stack_name *
_output_shapes
:

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnter;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid
„
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
T0
Ō
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid* 
_output_shapes
:

§
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1MulUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape* 
_output_shapes
:
*
T0
Į
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ń
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*(
_output_shapes
:’’’’’’’’’*
Tshape0
Æ
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
±
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
ē
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_accStack*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
_output_shapes
:*

stack_name 

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnter8encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh^gradients/Add*
_output_shapes
:*
swap_memory( *K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
T0
¢
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
T0
Ń
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh* 
_output_shapes
:

©
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mulMulRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop* 
_output_shapes
:
*
T0
Į
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0
ī
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1^gradients/Add*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
_output_shapes
:*
swap_memory( 
«
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
T0
Ś
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1* 
_output_shapes
:

­
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1* 
_output_shapes
:
*
T0
Ē
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ø
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*
T0*
Tshape0* 
_output_shapes
:

·
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape* 
_output_shapes
:
*
T0

gradients/AddN_6AddN2gradients/encoder_1/rnn/while/Select_1_grad/SelectPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1* 
_output_shapes
:
*
N*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_1_grad/Select*
T0
½
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape*
T0* 
_output_shapes
:

²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1*
T0* 
_output_shapes
:

­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
”
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
_output_shapes
: *
dtype0*
valueB 
Š
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Ē
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumSumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¬
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape*
T0*
Tshape0* 
_output_shapes
:

Ė
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1SumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ø
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0

;gradients/encoder_1/rnn/while/Switch_2_grad_1/NextIterationNextIterationgradients/AddN_6*
T0* 
_output_shapes
:

õ
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_accStack*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*

stack_name *
_output_shapes
:
 
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
£
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPush	StackPushXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnterCencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim^gradients/Add*
_output_shapes
:*
swap_memory( *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
T0
³
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc*
T0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
Ų
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPopStackPopagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop/RefEnter^gradients/Sub*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
_output_shapes
: 
Ė
Ogradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concatConcatV2Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop* 
_output_shapes
:
*
N*
T0*

Tidx0
ó
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
data_formatNHWC*
T0*
_output_shapes	
:
Ą
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
č
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMulMatMulOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
T0* 
_output_shapes
:
*
transpose_a( 

bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
»
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
æ
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat^gradients/Add*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat*
_output_shapes
:*
swap_memory( *
T0
Ī
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
ż
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat* 
_output_shapes
:

ļ
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1MatMulegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
transpose_b( * 
_output_shapes
:
*
transpose_a(*
T0
„
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
_output_shapes	
:*
dtype0
Ń
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc*
parallel_iterations *
T0*
_output_shapes	
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
Ģ
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:: 
ż
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
::
“
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/AddAddYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch:1Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
ė
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
ß
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:*
T0
Ŗ
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/RankConst^gradients/Sub*
_output_shapes
: *
dtype0*
value	B :

]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis
¶
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis
æ
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPush	StackPush`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnterIencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis^gradients/Add*
T0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*
swap_memory( 
É
igradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis
ī
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopStackPopigradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnter^gradients/Sub*
	elem_type0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
_output_shapes
: 
Ą
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/modFloorMod`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
ŗ
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
¹
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Shape_1Shapeencoder_1/rnn/while/Identity_3*
_output_shapes
:*
out_type0*
T0
Ų
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *.
_class$
" loc:@encoder_1/rnn/TensorArray_1

cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
T0*
is_constant(
ó
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPush	StackPushcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter%encoder_1/rnn/while/TensorArrayReadV3^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *.
_class$
" loc:@encoder_1/rnn/TensorArray_1
”
lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
Š
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopStackPoplgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnter^gradients/Sub*
	elem_type0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1* 
_output_shapes
:

Ī
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeNShapeNcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop*
N*
T0* 
_output_shapes
::*
out_type0
®
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffsetConcatOffsetWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/modZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
“
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/SliceSliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffsetZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN*
Index0*
T0* 
_output_shapes
:

Ā
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Slice_1SliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMulbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffset:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN:1*(
_output_shapes
:’’’’’’’’’*
Index0*
T0
ø
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_accConst* 
_output_shapes
:
*
dtype0*
valueB
*    
č
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1Enter_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations * 
_output_shapes
:
*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
ģ
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2Mergeagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIteration*"
_output_shapes
:
: *
N*
T0

`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/SwitchSwitchagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_output_shapes
:
:
*
T0
Ń
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/AddAddbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:


ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIterationNextIteration]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:

ö
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3Exit`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
*
T0
É
\gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterencoder_1/rnn/TensorArray_1*
is_constant(*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
ō
^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterHencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
is_constant(*
T0*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
 
Vgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3\gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*
_output_shapes

::*
source	gradients*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
č
Rgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1W^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: 
ł
^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity
­
agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnterRefEnter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

bgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPush	StackPushagradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnterencoder_1/rnn/while/Identity^gradients/Add*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
:*
swap_memory( *
T0
Ą
jgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop/RefEnterRefEnter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
T0*
is_constant(
å
agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopStackPopjgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop/RefEnter^gradients/Sub*
	elem_type0*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
: 
©
Xgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Vgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/SliceRgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: *
T0
¤
gradients/AddN_7AddN2gradients/encoder_1/rnn/while/Select_2_grad/Select[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Slice_1*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_2_grad/Select* 
_output_shapes
:
*
T0*
N

Bgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterBgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
parallel_iterations *
T0*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 

Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Jgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
N*
T0*
_output_shapes
: : 
Ė
Cgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*
_output_shapes
: : *
T0

@gradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddEgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Xgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
¾
Jgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration@gradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
²
Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitCgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
_output_shapes
: *
T0

;gradients/encoder_1/rnn/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_7* 
_output_shapes
:
*
T0
Ų
ygradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3encoder_1/rnn/TensorArray_1Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes

::*
source	gradients*.
_class$
" loc:@encoder_1/rnn/TensorArray_1

ugradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3z^gradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: 

kgradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3ygradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3&encoder_1/rnn/TensorArrayUnstack/rangeugradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*%
_output_shapes
:*
dtype0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
element_shape:

4gradients/encoder_1/transpose_grad/InvertPermutationInvertPermutationencoder_1/transpose/perm*
T0*
_output_shapes
:

,gradients/encoder_1/transpose_grad/transpose	Transposekgradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV34gradients/encoder_1/transpose_grad/InvertPermutation*
Tperm0*
T0*%
_output_shapes
:

+gradients/encoder/conv1d/Squeeze_grad/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
Õ
-gradients/encoder/conv1d/Squeeze_grad/ReshapeReshape,gradients/encoder_1/transpose_grad/transpose+gradients/encoder/conv1d/Squeeze_grad/Shape*
Tshape0*)
_output_shapes
:*
T0

*gradients/encoder/conv1d/Conv2D_grad/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
Ņ
8gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/encoder/conv1d/Conv2D_grad/Shapeencoder/conv1d/ExpandDims_1-gradients/encoder/conv1d/Squeeze_grad/Reshape*
use_cudnn_on_gpu(*
T0*
paddingVALID*(
_output_shapes
:*
data_formatNHWC*
strides


,gradients/encoder/conv1d/Conv2D_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            
Ó
9gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/conv1d/ExpandDims,gradients/encoder/conv1d/Conv2D_grad/Shape_1-gradients/encoder/conv1d/Squeeze_grad/Reshape*
paddingVALID*
T0*
data_formatNHWC*
strides
*'
_output_shapes
:*
use_cudnn_on_gpu(

0gradients/encoder/conv1d/ExpandDims_1_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         
ę
2gradients/encoder/conv1d/ExpandDims_1_grad/ReshapeReshape9gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropFilter0gradients/encoder/conv1d/ExpandDims_1_grad/Shape*#
_output_shapes
:*
Tshape0*
T0
ø
global_norm/L2LossL2Loss2gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*
_output_shapes
: *E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*
T0
ŗ
global_norm/L2Loss_1L2Loss2gradients/encoder_1/initial_state_0_tiled_grad/Sum*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
: 
ŗ
global_norm/L2Loss_2L2Loss2gradients/encoder_1/initial_state_1_tiled_grad/Sum*
T0*
_output_shapes
: *E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum
ŗ
global_norm/L2Loss_3L2Loss2gradients/encoder_1/initial_state_2_tiled_grad/Sum*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum*
_output_shapes
: *
T0
ŗ
global_norm/L2Loss_4L2Loss2gradients/encoder_1/initial_state_3_tiled_grad/Sum*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
_output_shapes
: *
T0

global_norm/L2Loss_5L2Lossagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: 

global_norm/L2Loss_6L2LossXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: 

global_norm/L2Loss_7L2Lossagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *
T0

global_norm/L2Loss_8L2LossXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: *k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
Ź
global_norm/L2Loss_9L2Loss:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
T0*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
_output_shapes
: 
Ć
global_norm/L2Loss_10L2Loss6gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
T0*
_output_shapes
: *I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad
Ä
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3global_norm/L2Loss_4global_norm/L2Loss_5global_norm/L2Loss_6global_norm/L2Loss_7global_norm/L2Loss_8global_norm/L2Loss_9global_norm/L2Loss_10*
_output_shapes
:*
N*

axis *
T0
[
global_norm/ConstConst*
valueB: *
_output_shapes
:*
dtype0
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
X
global_norm/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *   @
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
_output_shapes
: *
T0
Q
global_norm/global_normSqrtglobal_norm/mul*
_output_shapes
: *
T0
b
clip_by_global_norm/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xglobal_norm/global_norm*
_output_shapes
: *
T0
^
clip_by_global_norm/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
d
clip_by_global_norm/truediv_1/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0

clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 

clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
_output_shapes
: *
T0
^
clip_by_global_norm/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
T0*
_output_shapes
: 
ā
clip_by_global_norm/mul_1Mul2gradients/encoder/conv1d/ExpandDims_1_grad/Reshapeclip_by_global_norm/mul*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*#
_output_shapes
:*
T0
Ę
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*
T0*#
_output_shapes
:*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape
Ž
clip_by_global_norm/mul_2Mul2gradients/encoder_1/initial_state_0_tiled_grad/Sumclip_by_global_norm/mul*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
:	
Ā
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
_output_shapes
:	*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
T0
Ž
clip_by_global_norm/mul_3Mul2gradients/encoder_1/initial_state_1_tiled_grad/Sumclip_by_global_norm/mul*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
_output_shapes
:	*
T0
Ā
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3*
T0*
_output_shapes
:	*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum
Ž
clip_by_global_norm/mul_4Mul2gradients/encoder_1/initial_state_2_tiled_grad/Sumclip_by_global_norm/mul*
T0*
_output_shapes
:	*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum
Ā
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum*
_output_shapes
:	
Ž
clip_by_global_norm/mul_5Mul2gradients/encoder_1/initial_state_3_tiled_grad/Sumclip_by_global_norm/mul*
T0*
_output_shapes
:	*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum
Ā
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
_output_shapes
:	*
T0
½
clip_by_global_norm/mul_6Mulagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
*
T0
ņ
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6* 
_output_shapes
:
*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
¦
clip_by_global_norm/mul_7MulXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
_output_shapes	
:*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
ä
*clip_by_global_norm/clip_by_global_norm/_6Identityclip_by_global_norm/mul_7*
_output_shapes	
:*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
½
clip_by_global_norm/mul_8Mulagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
*
T0
ņ
*clip_by_global_norm/clip_by_global_norm/_7Identityclip_by_global_norm/mul_8*
T0*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:

¦
clip_by_global_norm/mul_9MulXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:*
T0
ä
*clip_by_global_norm/clip_by_global_norm/_8Identityclip_by_global_norm/mul_9*
T0*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:
ļ
clip_by_global_norm/mul_10Mul:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1clip_by_global_norm/mul*
_output_shapes
:	*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
T0
Ė
*clip_by_global_norm/clip_by_global_norm/_9Identityclip_by_global_norm/mul_10*
T0*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
_output_shapes
:	
ā
clip_by_global_norm/mul_11Mul6gradients/output_projection/xw_plus_b_grad/BiasAddGradclip_by_global_norm/mul*I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
_output_shapes
:*
T0
Ć
+clip_by_global_norm/clip_by_global_norm/_10Identityclip_by_global_norm/mul_11*I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
_output_shapes
:*
T0
p
grad_norms/grad_norms/tagsConst*
dtype0*
_output_shapes
: *&
valueB Bgrad_norms/grad_norms
|
grad_norms/grad_normsScalarSummarygrad_norms/grad_norms/tagsglobal_norm/global_norm*
_output_shapes
: *
T0
~
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@input_embed*
_output_shapes
: *
dtype0

beta1_power
VariableV2*
shared_name *
shape: *
_output_shapes
: *
_class
loc:@input_embed*
dtype0*
	container 
®
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@input_embed
j
beta1_power/readIdentitybeta1_power*
_class
loc:@input_embed*
_output_shapes
: *
T0
~
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w¾?*
_class
loc:@input_embed

beta2_power
VariableV2*
	container *
shared_name *
dtype0*
shape: *
_output_shapes
: *
_class
loc:@input_embed
®
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
j
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@input_embed*
_output_shapes
: 
d
zerosConst*#
_output_shapes
:*
dtype0*"
valueB*    
®
input_embed/Adam
VariableV2*
shared_name *
_class
loc:@input_embed*
	container *
shape:*
dtype0*#
_output_shapes
:
±
input_embed/Adam/AssignAssigninput_embed/Adamzeros*#
_output_shapes
:*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(

input_embed/Adam/readIdentityinput_embed/Adam*
T0*#
_output_shapes
:*
_class
loc:@input_embed
f
zeros_1Const*"
valueB*    *#
_output_shapes
:*
dtype0
°
input_embed/Adam_1
VariableV2*
shared_name *
shape:*#
_output_shapes
:*
_class
loc:@input_embed*
dtype0*
	container 
·
input_embed/Adam_1/AssignAssigninput_embed/Adam_1zeros_1*
use_locking(*
validate_shape(*
T0*#
_output_shapes
:*
_class
loc:@input_embed

input_embed/Adam_1/readIdentityinput_embed/Adam_1*#
_output_shapes
:*
_class
loc:@input_embed*
T0
^
zeros_2Const*
valueB	*    *
_output_shapes
:	*
dtype0
¾
encoder/initial_state_0/Adam
VariableV2*
shape:	*
_output_shapes
:	*
shared_name **
_class 
loc:@encoder/initial_state_0*
dtype0*
	container 
Ó
#encoder/initial_state_0/Adam/AssignAssignencoder/initial_state_0/Adamzeros_2*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(
”
!encoder/initial_state_0/Adam/readIdentityencoder/initial_state_0/Adam**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
T0
^
zeros_3Const*
_output_shapes
:	*
dtype0*
valueB	*    
Ą
encoder/initial_state_0/Adam_1
VariableV2*
_output_shapes
:	*
dtype0*
shape:	*
	container **
_class 
loc:@encoder/initial_state_0*
shared_name 
×
%encoder/initial_state_0/Adam_1/AssignAssignencoder/initial_state_0/Adam_1zeros_3*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_0*
validate_shape(*
_output_shapes
:	
„
#encoder/initial_state_0/Adam_1/readIdentityencoder/initial_state_0/Adam_1*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0
^
zeros_4Const*
dtype0*
_output_shapes
:	*
valueB	*    
¾
encoder/initial_state_1/Adam
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_1*
shared_name *
_output_shapes
:	*
shape:	
Ó
#encoder/initial_state_1/Adam/AssignAssignencoder/initial_state_1/Adamzeros_4**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
”
!encoder/initial_state_1/Adam/readIdentityencoder/initial_state_1/Adam*
T0**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	
^
zeros_5Const*
dtype0*
_output_shapes
:	*
valueB	*    
Ą
encoder/initial_state_1/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
shape:	*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1
×
%encoder/initial_state_1/Adam_1/AssignAssignencoder/initial_state_1/Adam_1zeros_5*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	
„
#encoder/initial_state_1/Adam_1/readIdentityencoder/initial_state_1/Adam_1*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1*
T0
^
zeros_6Const*
valueB	*    *
dtype0*
_output_shapes
:	
¾
encoder/initial_state_2/Adam
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_2*
shared_name *
_output_shapes
:	*
shape:	
Ó
#encoder/initial_state_2/Adam/AssignAssignencoder/initial_state_2/Adamzeros_6*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2
”
!encoder/initial_state_2/Adam/readIdentityencoder/initial_state_2/Adam**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	*
T0
^
zeros_7Const*
_output_shapes
:	*
dtype0*
valueB	*    
Ą
encoder/initial_state_2/Adam_1
VariableV2**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	*
shape:	*
dtype0*
shared_name *
	container 
×
%encoder/initial_state_2/Adam_1/AssignAssignencoder/initial_state_2/Adam_1zeros_7*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_2*
validate_shape(*
_output_shapes
:	
„
#encoder/initial_state_2/Adam_1/readIdentityencoder/initial_state_2/Adam_1*
T0**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	
^
zeros_8Const*
valueB	*    *
_output_shapes
:	*
dtype0
¾
encoder/initial_state_3/Adam
VariableV2*
shared_name *
shape:	*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3*
dtype0*
	container 
Ó
#encoder/initial_state_3/Adam/AssignAssignencoder/initial_state_3/Adamzeros_8*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	
”
!encoder/initial_state_3/Adam/readIdentityencoder/initial_state_3/Adam*
T0**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	
^
zeros_9Const*
valueB	*    *
dtype0*
_output_shapes
:	
Ą
encoder/initial_state_3/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
shape:	*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3
×
%encoder/initial_state_3/Adam_1/AssignAssignencoder/initial_state_3/Adam_1zeros_9*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	
„
#encoder/initial_state_3/Adam_1/readIdentityencoder/initial_state_3/Adam_1*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3*
T0
a
zeros_10Const*
dtype0* 
_output_shapes
:
*
valueB
*    
ų
8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam
VariableV2* 
_output_shapes
:
*
dtype0*
shape:
*
	container *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
shared_name 
©
?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/AssignAssign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamzeros_10*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
validate_shape(* 
_output_shapes
:

ö
=encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/readIdentity8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:

a
zeros_11Const*
dtype0* 
_output_shapes
:
*
valueB
*    
ś
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1
VariableV2*
	container *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
shape:
*
shared_name 
­
Aencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/AssignAssign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1zeros_11*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
ś
?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/readIdentity:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0
W
zeros_12Const*
valueB*    *
_output_shapes	
:*
dtype0
ģ
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam
VariableV2*
shape:*
_output_shapes	
:*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
dtype0*
	container 
”
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adamzeros_12*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
ī
<encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/readIdentity7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
T0
W
zeros_13Const*
valueB*    *
_output_shapes	
:*
dtype0
ī
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1
VariableV2*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
shape:*
dtype0*
shared_name *
	container 
„
@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1zeros_13*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
ņ
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/readIdentity9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
a
zeros_14Const*
valueB
*    *
dtype0* 
_output_shapes
:

ų
8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam
VariableV2*
shared_name *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
	container *
shape:
*
dtype0* 
_output_shapes
:

©
?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/AssignAssign8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adamzeros_14* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0*
use_locking(
ö
=encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/readIdentity8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:

a
zeros_15Const*
dtype0* 
_output_shapes
:
*
valueB
*    
ś
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1
VariableV2*
	container *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
shape:
*
shared_name 
­
Aencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/AssignAssign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1zeros_15*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
ś
?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/readIdentity:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
T0
W
zeros_16Const*
dtype0*
_output_shapes	
:*
valueB*    
ģ
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam
VariableV2*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:*
shape:*
dtype0*
shared_name *
	container 
”
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adamzeros_16*
_output_shapes	
:*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
use_locking(
ī
<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/readIdentity7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:*
T0
W
zeros_17Const*
dtype0*
_output_shapes	
:*
valueB*    
ī
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1
VariableV2*
	container *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
shared_name *
_output_shapes	
:*
shape:
„
@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1zeros_17*
_output_shapes	
:*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
use_locking(
ņ
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/readIdentity9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:
_
zeros_18Const*
dtype0*
_output_shapes
:	*
valueB	*    
¶
output_projection/W/Adam
VariableV2*
shape:	*
_output_shapes
:	*
shared_name *&
_class
loc:@output_projection/W*
dtype0*
	container 
Č
output_projection/W/Adam/AssignAssignoutput_projection/W/Adamzeros_18*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

output_projection/W/Adam/readIdentityoutput_projection/W/Adam*
T0*
_output_shapes
:	*&
_class
loc:@output_projection/W
_
zeros_19Const*
_output_shapes
:	*
dtype0*
valueB	*    
ø
output_projection/W/Adam_1
VariableV2*
	container *
dtype0*&
_class
loc:@output_projection/W*
_output_shapes
:	*
shape:	*
shared_name 
Ģ
!output_projection/W/Adam_1/AssignAssignoutput_projection/W/Adam_1zeros_19*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

output_projection/W/Adam_1/readIdentityoutput_projection/W/Adam_1*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0
U
zeros_20Const*
valueB*    *
_output_shapes
:*
dtype0
¬
output_projection/b/Adam
VariableV2*
	container *
dtype0*&
_class
loc:@output_projection/b*
shared_name *
_output_shapes
:*
shape:
Ć
output_projection/b/Adam/AssignAssignoutput_projection/b/Adamzeros_20*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*&
_class
loc:@output_projection/b

output_projection/b/Adam/readIdentityoutput_projection/b/Adam*
T0*&
_class
loc:@output_projection/b*
_output_shapes
:
U
zeros_21Const*
_output_shapes
:*
dtype0*
valueB*    
®
output_projection/b/Adam_1
VariableV2*
shared_name *
shape:*
_output_shapes
:*&
_class
loc:@output_projection/b*
dtype0*
	container 
Ē
!output_projection/b/Adam_1/AssignAssignoutput_projection/b/Adam_1zeros_21*
use_locking(*
T0*&
_class
loc:@output_projection/b*
validate_shape(*
_output_shapes
:

output_projection/b/Adam_1/readIdentityoutput_projection/b/Adam_1*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0
O

Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
O

Adam/beta2Const*
valueB
 *w¾?*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
valueB
 *wĢ+2*
_output_shapes
: *
dtype0
Ē
!Adam/update_input_embed/ApplyAdam	ApplyAdaminput_embedinput_embed/Adaminput_embed/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
_class
loc:@input_embed*#
_output_shapes
:*
T0*
use_locking( 
’
-Adam/update_encoder/initial_state_0/ApplyAdam	ApplyAdamencoder/initial_state_0encoder/initial_state_0/Adamencoder/initial_state_0/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_locking( *
T0**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	
’
-Adam/update_encoder/initial_state_1/ApplyAdam	ApplyAdamencoder/initial_state_1encoder/initial_state_1/Adamencoder/initial_state_1/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
use_locking( *
T0**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	
’
-Adam/update_encoder/initial_state_2/ApplyAdam	ApplyAdamencoder/initial_state_2encoder/initial_state_2/Adamencoder/initial_state_2/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	*
T0*
use_locking( 
’
-Adam/update_encoder/initial_state_3/ApplyAdam	ApplyAdamencoder/initial_state_3encoder/initial_state_3/Adamencoder/initial_state_3/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_4*
use_locking( *
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3

IAdam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdam	ApplyAdam3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_5*
use_locking( *
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights

HAdam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdam	ApplyAdam2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_6*
use_locking( *
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:

IAdam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdam	ApplyAdam3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_7*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
T0*
use_locking( 

HAdam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam	ApplyAdam2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_8*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:*
T0*
use_locking( 
ė
)Adam/update_output_projection/W/ApplyAdam	ApplyAdamoutput_projection/Woutput_projection/W/Adamoutput_projection/W/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_9*
_output_shapes
:	*&
_class
loc:@output_projection/W*
T0*
use_locking( 
ē
)Adam/update_output_projection/b/ApplyAdam	ApplyAdamoutput_projection/boutput_projection/b/Adamoutput_projection/b/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_10*
_output_shapes
:*&
_class
loc:@output_projection/b*
T0*
use_locking( 
Ų
Adam/mulMulbeta1_power/read
Adam/beta1"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@input_embed

Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking( 
Ś

Adam/mul_1Mulbeta2_power/read
Adam/beta2"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam*
T0*
_class
loc:@input_embed*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_class
loc:@input_embed*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 

Adam/updateNoOp"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam^Adam/Assign^Adam/Assign_1
w

Adam/valueConst^Adam/update*
dtype0*
_output_shapes
: *
value	B :*
_class
loc:@Variable
x
Adam	AssignAddVariable
Adam/value*
use_locking( *
T0*
_output_shapes
: *
_class
loc:@Variable
Ö
Const_1Const*
dtype0	*
_output_shapes	
:*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
b
ArgMaxArgMaxcond/Merge_1ArgMax/dimension*

Tidx0*
T0*
_output_shapes	
:
E
EqualEqualArgMaxConst_1*
_output_shapes	
:*
T0	
L

LogicalAnd
LogicalAndaccuracy/EqualEqual*
_output_shapes	
:
Z
count_nonzero/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
n
count_nonzero/NotEqualNotEqual
LogicalAndcount_nonzero/NotEqual/y*
T0
*
_output_shapes	
:
j
count_nonzero/ToInt64Castcount_nonzero/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

]
count_nonzero/ConstConst*
valueB: *
dtype0*
_output_shapes
:

count_nonzero/SumSumcount_nonzero/ToInt64count_nonzero/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
Y
Equal_1Equaloutput_projection/ArgMaxConst_1*
T0	*
_output_shapes	
:
\
count_nonzero_1/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
o
count_nonzero_1/NotEqualNotEqualEqual_1count_nonzero_1/NotEqual/y*
_output_shapes	
:*
T0

n
count_nonzero_1/ToInt64Castcount_nonzero_1/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

_
count_nonzero_1/ConstConst*
valueB: *
_output_shapes
:*
dtype0

count_nonzero_1/SumSumcount_nonzero_1/ToInt64count_nonzero_1/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
f
ArgMax_1ArgMaxcond/Merge_1ArgMax_1/dimension*
_output_shapes	
:*
T0*

Tidx0
I
Equal_2EqualArgMax_1Const_1*
T0	*
_output_shapes	
:
\
count_nonzero_2/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
o
count_nonzero_2/NotEqualNotEqualEqual_2count_nonzero_2/NotEqual/y*
_output_shapes	
:*
T0

n
count_nonzero_2/ToInt64Castcount_nonzero_2/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

_
count_nonzero_2/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

count_nonzero_2/SumSumcount_nonzero_2/ToInt64count_nonzero_2/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
K
	Greater/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
S
GreaterGreatercount_nonzero_1/Sum	Greater/y*
T0	*
_output_shapes
: 
L
cond_1/SwitchSwitchGreaterGreater*
_output_shapes
: : *
T0

M
cond_1/switch_tIdentitycond_1/Switch:1*
T0
*
_output_shapes
: 
K
cond_1/switch_fIdentitycond_1/Switch*
T0
*
_output_shapes
: 
D
cond_1/pred_idIdentityGreater*
_output_shapes
: *
T0


cond_1/truediv/Cast/SwitchSwitchcount_nonzero/Sumcond_1/pred_id*
T0	*$
_class
loc:@count_nonzero/Sum*
_output_shapes
: : 
i
cond_1/truediv/CastCastcond_1/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	

cond_1/truediv/Cast_1/SwitchSwitchcount_nonzero_1/Sumcond_1/pred_id*&
_class
loc:@count_nonzero_1/Sum*
_output_shapes
: : *
T0	
m
cond_1/truediv/Cast_1Castcond_1/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
f
cond_1/truedivRealDivcond_1/truediv/Castcond_1/truediv/Cast_1*
_output_shapes
: *
T0
g
cond_1/ConstConst^cond_1/switch_f*
valueB 2        *
_output_shapes
: *
dtype0
_
cond_1/MergeMergecond_1/Constcond_1/truediv*
_output_shapes
: : *
N*
T0
\
precision_0/tagsConst*
valueB Bprecision_0*
dtype0*
_output_shapes
: 
]
precision_0ScalarSummaryprecision_0/tagscond_1/Merge*
_output_shapes
: *
T0
M
Greater_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
W
	Greater_1Greatercount_nonzero_2/SumGreater_1/y*
T0	*
_output_shapes
: 
P
cond_2/SwitchSwitch	Greater_1	Greater_1*
T0
*
_output_shapes
: : 
M
cond_2/switch_tIdentitycond_2/Switch:1*
T0
*
_output_shapes
: 
K
cond_2/switch_fIdentitycond_2/Switch*
_output_shapes
: *
T0

F
cond_2/pred_idIdentity	Greater_1*
_output_shapes
: *
T0


cond_2/truediv/Cast/SwitchSwitchcount_nonzero/Sumcond_2/pred_id*
_output_shapes
: : *$
_class
loc:@count_nonzero/Sum*
T0	
i
cond_2/truediv/CastCastcond_2/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_2/truediv/Cast_1/SwitchSwitchcount_nonzero_2/Sumcond_2/pred_id*
T0	*&
_class
loc:@count_nonzero_2/Sum*
_output_shapes
: : 
m
cond_2/truediv/Cast_1Castcond_2/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
f
cond_2/truedivRealDivcond_2/truediv/Castcond_2/truediv/Cast_1*
T0*
_output_shapes
: 
g
cond_2/ConstConst^cond_2/switch_f*
valueB 2        *
dtype0*
_output_shapes
: 
_
cond_2/MergeMergecond_2/Constcond_2/truediv*
_output_shapes
: : *
N*
T0
V
recall_0/tagsConst*
valueB Brecall_0*
dtype0*
_output_shapes
: 
W
recall_0ScalarSummaryrecall_0/tagscond_2/Merge*
_output_shapes
: *
T0
Ö
Const_2Const*
_output_shapes	
:*
dtype0	*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
T
ArgMax_2/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
f
ArgMax_2ArgMaxcond/Merge_1ArgMax_2/dimension*

Tidx0*
T0*
_output_shapes	
:
I
Equal_3EqualArgMax_2Const_2*
_output_shapes	
:*
T0	
P
LogicalAnd_1
LogicalAndaccuracy/EqualEqual_3*
_output_shapes	
:
\
count_nonzero_3/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

t
count_nonzero_3/NotEqualNotEqualLogicalAnd_1count_nonzero_3/NotEqual/y*
_output_shapes	
:*
T0

n
count_nonzero_3/ToInt64Castcount_nonzero_3/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

_
count_nonzero_3/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

count_nonzero_3/SumSumcount_nonzero_3/ToInt64count_nonzero_3/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
Y
Equal_4Equaloutput_projection/ArgMaxConst_2*
T0	*
_output_shapes	
:
\
count_nonzero_4/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
o
count_nonzero_4/NotEqualNotEqualEqual_4count_nonzero_4/NotEqual/y*
_output_shapes	
:*
T0

n
count_nonzero_4/ToInt64Castcount_nonzero_4/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
_
count_nonzero_4/ConstConst*
valueB: *
dtype0*
_output_shapes
:

count_nonzero_4/SumSumcount_nonzero_4/ToInt64count_nonzero_4/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
T
ArgMax_3/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
f
ArgMax_3ArgMaxcond/Merge_1ArgMax_3/dimension*
_output_shapes	
:*
T0*

Tidx0
I
Equal_5EqualArgMax_3Const_2*
_output_shapes	
:*
T0	
\
count_nonzero_5/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
o
count_nonzero_5/NotEqualNotEqualEqual_5count_nonzero_5/NotEqual/y*
_output_shapes	
:*
T0

n
count_nonzero_5/ToInt64Castcount_nonzero_5/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
_
count_nonzero_5/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

count_nonzero_5/SumSumcount_nonzero_5/ToInt64count_nonzero_5/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
M
Greater_2/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
W
	Greater_2Greatercount_nonzero_4/SumGreater_2/y*
T0	*
_output_shapes
: 
P
cond_3/SwitchSwitch	Greater_2	Greater_2*
T0
*
_output_shapes
: : 
M
cond_3/switch_tIdentitycond_3/Switch:1*
T0
*
_output_shapes
: 
K
cond_3/switch_fIdentitycond_3/Switch*
_output_shapes
: *
T0

F
cond_3/pred_idIdentity	Greater_2*
_output_shapes
: *
T0


cond_3/truediv/Cast/SwitchSwitchcount_nonzero_3/Sumcond_3/pred_id*
T0	*
_output_shapes
: : *&
_class
loc:@count_nonzero_3/Sum
i
cond_3/truediv/CastCastcond_3/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_3/truediv/Cast_1/SwitchSwitchcount_nonzero_4/Sumcond_3/pred_id*
_output_shapes
: : *&
_class
loc:@count_nonzero_4/Sum*
T0	
m
cond_3/truediv/Cast_1Castcond_3/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
f
cond_3/truedivRealDivcond_3/truediv/Castcond_3/truediv/Cast_1*
T0*
_output_shapes
: 
g
cond_3/ConstConst^cond_3/switch_f*
dtype0*
_output_shapes
: *
valueB 2        
_
cond_3/MergeMergecond_3/Constcond_3/truediv*
N*
T0*
_output_shapes
: : 
\
precision_1/tagsConst*
dtype0*
_output_shapes
: *
valueB Bprecision_1
]
precision_1ScalarSummaryprecision_1/tagscond_3/Merge*
_output_shapes
: *
T0
M
Greater_3/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
W
	Greater_3Greatercount_nonzero_5/SumGreater_3/y*
_output_shapes
: *
T0	
P
cond_4/SwitchSwitch	Greater_3	Greater_3*
_output_shapes
: : *
T0

M
cond_4/switch_tIdentitycond_4/Switch:1*
_output_shapes
: *
T0

K
cond_4/switch_fIdentitycond_4/Switch*
_output_shapes
: *
T0

F
cond_4/pred_idIdentity	Greater_3*
T0
*
_output_shapes
: 

cond_4/truediv/Cast/SwitchSwitchcount_nonzero_3/Sumcond_4/pred_id*
T0	*
_output_shapes
: : *&
_class
loc:@count_nonzero_3/Sum
i
cond_4/truediv/CastCastcond_4/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	

cond_4/truediv/Cast_1/SwitchSwitchcount_nonzero_5/Sumcond_4/pred_id*&
_class
loc:@count_nonzero_5/Sum*
_output_shapes
: : *
T0	
m
cond_4/truediv/Cast_1Castcond_4/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
f
cond_4/truedivRealDivcond_4/truediv/Castcond_4/truediv/Cast_1*
T0*
_output_shapes
: 
g
cond_4/ConstConst^cond_4/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
_
cond_4/MergeMergecond_4/Constcond_4/truediv*
N*
T0*
_output_shapes
: : 
V
recall_1/tagsConst*
valueB Brecall_1*
dtype0*
_output_shapes
: 
W
recall_1ScalarSummaryrecall_1/tagscond_4/Merge*
T0*
_output_shapes
: 
Ö
Const_3Const*
dtype0	*
_output_shapes	
:*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
T
ArgMax_4/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
f
ArgMax_4ArgMaxcond/Merge_1ArgMax_4/dimension*

Tidx0*
T0*
_output_shapes	
:
I
Equal_6EqualArgMax_4Const_3*
T0	*
_output_shapes	
:
P
LogicalAnd_2
LogicalAndaccuracy/EqualEqual_6*
_output_shapes	
:
\
count_nonzero_6/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

t
count_nonzero_6/NotEqualNotEqualLogicalAnd_2count_nonzero_6/NotEqual/y*
T0
*
_output_shapes	
:
n
count_nonzero_6/ToInt64Castcount_nonzero_6/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
_
count_nonzero_6/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

count_nonzero_6/SumSumcount_nonzero_6/ToInt64count_nonzero_6/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
Y
Equal_7Equaloutput_projection/ArgMaxConst_3*
T0	*
_output_shapes	
:
\
count_nonzero_7/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
o
count_nonzero_7/NotEqualNotEqualEqual_7count_nonzero_7/NotEqual/y*
T0
*
_output_shapes	
:
n
count_nonzero_7/ToInt64Castcount_nonzero_7/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

_
count_nonzero_7/ConstConst*
valueB: *
_output_shapes
:*
dtype0

count_nonzero_7/SumSumcount_nonzero_7/ToInt64count_nonzero_7/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
T
ArgMax_5/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
f
ArgMax_5ArgMaxcond/Merge_1ArgMax_5/dimension*

Tidx0*
T0*
_output_shapes	
:
I
Equal_8EqualArgMax_5Const_3*
_output_shapes	
:*
T0	
\
count_nonzero_8/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
o
count_nonzero_8/NotEqualNotEqualEqual_8count_nonzero_8/NotEqual/y*
T0
*
_output_shapes	
:
n
count_nonzero_8/ToInt64Castcount_nonzero_8/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

_
count_nonzero_8/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

count_nonzero_8/SumSumcount_nonzero_8/ToInt64count_nonzero_8/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
M
Greater_4/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
W
	Greater_4Greatercount_nonzero_7/SumGreater_4/y*
T0	*
_output_shapes
: 
P
cond_5/SwitchSwitch	Greater_4	Greater_4*
_output_shapes
: : *
T0

M
cond_5/switch_tIdentitycond_5/Switch:1*
_output_shapes
: *
T0

K
cond_5/switch_fIdentitycond_5/Switch*
_output_shapes
: *
T0

F
cond_5/pred_idIdentity	Greater_4*
T0
*
_output_shapes
: 

cond_5/truediv/Cast/SwitchSwitchcount_nonzero_6/Sumcond_5/pred_id*
T0	*&
_class
loc:@count_nonzero_6/Sum*
_output_shapes
: : 
i
cond_5/truediv/CastCastcond_5/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	

cond_5/truediv/Cast_1/SwitchSwitchcount_nonzero_7/Sumcond_5/pred_id*
_output_shapes
: : *&
_class
loc:@count_nonzero_7/Sum*
T0	
m
cond_5/truediv/Cast_1Castcond_5/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
f
cond_5/truedivRealDivcond_5/truediv/Castcond_5/truediv/Cast_1*
_output_shapes
: *
T0
g
cond_5/ConstConst^cond_5/switch_f*
valueB 2        *
dtype0*
_output_shapes
: 
_
cond_5/MergeMergecond_5/Constcond_5/truediv*
_output_shapes
: : *
T0*
N
\
precision_2/tagsConst*
dtype0*
_output_shapes
: *
valueB Bprecision_2
]
precision_2ScalarSummaryprecision_2/tagscond_5/Merge*
_output_shapes
: *
T0
M
Greater_5/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
W
	Greater_5Greatercount_nonzero_8/SumGreater_5/y*
_output_shapes
: *
T0	
P
cond_6/SwitchSwitch	Greater_5	Greater_5*
T0
*
_output_shapes
: : 
M
cond_6/switch_tIdentitycond_6/Switch:1*
T0
*
_output_shapes
: 
K
cond_6/switch_fIdentitycond_6/Switch*
T0
*
_output_shapes
: 
F
cond_6/pred_idIdentity	Greater_5*
_output_shapes
: *
T0


cond_6/truediv/Cast/SwitchSwitchcount_nonzero_6/Sumcond_6/pred_id*
_output_shapes
: : *&
_class
loc:@count_nonzero_6/Sum*
T0	
i
cond_6/truediv/CastCastcond_6/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_6/truediv/Cast_1/SwitchSwitchcount_nonzero_8/Sumcond_6/pred_id*
_output_shapes
: : *&
_class
loc:@count_nonzero_8/Sum*
T0	
m
cond_6/truediv/Cast_1Castcond_6/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
f
cond_6/truedivRealDivcond_6/truediv/Castcond_6/truediv/Cast_1*
T0*
_output_shapes
: 
g
cond_6/ConstConst^cond_6/switch_f*
valueB 2        *
_output_shapes
: *
dtype0
_
cond_6/MergeMergecond_6/Constcond_6/truediv*
N*
T0*
_output_shapes
: : 
V
recall_2/tagsConst*
dtype0*
_output_shapes
: *
valueB Brecall_2
W
recall_2ScalarSummaryrecall_2/tagscond_6/Merge*
T0*
_output_shapes
: 
Ö
Const_4Const*
_output_shapes	
:*
dtype0	*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
T
ArgMax_6/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
f
ArgMax_6ArgMaxcond/Merge_1ArgMax_6/dimension*

Tidx0*
T0*
_output_shapes	
:
I
Equal_9EqualArgMax_6Const_4*
_output_shapes	
:*
T0	
P
LogicalAnd_3
LogicalAndaccuracy/EqualEqual_9*
_output_shapes	
:
\
count_nonzero_9/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
t
count_nonzero_9/NotEqualNotEqualLogicalAnd_3count_nonzero_9/NotEqual/y*
T0
*
_output_shapes	
:
n
count_nonzero_9/ToInt64Castcount_nonzero_9/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
_
count_nonzero_9/ConstConst*
valueB: *
dtype0*
_output_shapes
:

count_nonzero_9/SumSumcount_nonzero_9/ToInt64count_nonzero_9/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
Z
Equal_10Equaloutput_projection/ArgMaxConst_4*
_output_shapes	
:*
T0	
]
count_nonzero_10/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
r
count_nonzero_10/NotEqualNotEqualEqual_10count_nonzero_10/NotEqual/y*
_output_shapes	
:*
T0

p
count_nonzero_10/ToInt64Castcount_nonzero_10/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
`
count_nonzero_10/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

count_nonzero_10/SumSumcount_nonzero_10/ToInt64count_nonzero_10/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
T
ArgMax_7/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
f
ArgMax_7ArgMaxcond/Merge_1ArgMax_7/dimension*
_output_shapes	
:*
T0*

Tidx0
J
Equal_11EqualArgMax_7Const_4*
T0	*
_output_shapes	
:
]
count_nonzero_11/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
r
count_nonzero_11/NotEqualNotEqualEqual_11count_nonzero_11/NotEqual/y*
_output_shapes	
:*
T0

p
count_nonzero_11/ToInt64Castcount_nonzero_11/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

`
count_nonzero_11/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

count_nonzero_11/SumSumcount_nonzero_11/ToInt64count_nonzero_11/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
M
Greater_6/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
X
	Greater_6Greatercount_nonzero_10/SumGreater_6/y*
_output_shapes
: *
T0	
P
cond_7/SwitchSwitch	Greater_6	Greater_6*
T0
*
_output_shapes
: : 
M
cond_7/switch_tIdentitycond_7/Switch:1*
T0
*
_output_shapes
: 
K
cond_7/switch_fIdentitycond_7/Switch*
_output_shapes
: *
T0

F
cond_7/pred_idIdentity	Greater_6*
_output_shapes
: *
T0


cond_7/truediv/Cast/SwitchSwitchcount_nonzero_9/Sumcond_7/pred_id*&
_class
loc:@count_nonzero_9/Sum*
_output_shapes
: : *
T0	
i
cond_7/truediv/CastCastcond_7/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_7/truediv/Cast_1/SwitchSwitchcount_nonzero_10/Sumcond_7/pred_id*
T0	*'
_class
loc:@count_nonzero_10/Sum*
_output_shapes
: : 
m
cond_7/truediv/Cast_1Castcond_7/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
f
cond_7/truedivRealDivcond_7/truediv/Castcond_7/truediv/Cast_1*
T0*
_output_shapes
: 
g
cond_7/ConstConst^cond_7/switch_f*
valueB 2        *
_output_shapes
: *
dtype0
_
cond_7/MergeMergecond_7/Constcond_7/truediv*
T0*
N*
_output_shapes
: : 
\
precision_3/tagsConst*
valueB Bprecision_3*
dtype0*
_output_shapes
: 
]
precision_3ScalarSummaryprecision_3/tagscond_7/Merge*
_output_shapes
: *
T0
M
Greater_7/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
X
	Greater_7Greatercount_nonzero_11/SumGreater_7/y*
T0	*
_output_shapes
: 
P
cond_8/SwitchSwitch	Greater_7	Greater_7*
T0
*
_output_shapes
: : 
M
cond_8/switch_tIdentitycond_8/Switch:1*
T0
*
_output_shapes
: 
K
cond_8/switch_fIdentitycond_8/Switch*
_output_shapes
: *
T0

F
cond_8/pred_idIdentity	Greater_7*
_output_shapes
: *
T0


cond_8/truediv/Cast/SwitchSwitchcount_nonzero_9/Sumcond_8/pred_id*
T0	*
_output_shapes
: : *&
_class
loc:@count_nonzero_9/Sum
i
cond_8/truediv/CastCastcond_8/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_8/truediv/Cast_1/SwitchSwitchcount_nonzero_11/Sumcond_8/pred_id*'
_class
loc:@count_nonzero_11/Sum*
_output_shapes
: : *
T0	
m
cond_8/truediv/Cast_1Castcond_8/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
f
cond_8/truedivRealDivcond_8/truediv/Castcond_8/truediv/Cast_1*
T0*
_output_shapes
: 
g
cond_8/ConstConst^cond_8/switch_f*
dtype0*
_output_shapes
: *
valueB 2        
_
cond_8/MergeMergecond_8/Constcond_8/truediv*
N*
T0*
_output_shapes
: : 
V
recall_3/tagsConst*
dtype0*
_output_shapes
: *
valueB Brecall_3
W
recall_3ScalarSummaryrecall_3/tagscond_8/Merge*
_output_shapes
: *
T0
Ö
Const_5Const*
dtype0	*
_output_shapes	
:*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
T
ArgMax_8/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
f
ArgMax_8ArgMaxcond/Merge_1ArgMax_8/dimension*
_output_shapes	
:*
T0*

Tidx0
J
Equal_12EqualArgMax_8Const_5*
_output_shapes	
:*
T0	
Q
LogicalAnd_4
LogicalAndaccuracy/EqualEqual_12*
_output_shapes	
:
]
count_nonzero_12/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
v
count_nonzero_12/NotEqualNotEqualLogicalAnd_4count_nonzero_12/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_12/ToInt64Castcount_nonzero_12/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

`
count_nonzero_12/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

count_nonzero_12/SumSumcount_nonzero_12/ToInt64count_nonzero_12/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
Z
Equal_13Equaloutput_projection/ArgMaxConst_5*
T0	*
_output_shapes	
:
]
count_nonzero_13/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

r
count_nonzero_13/NotEqualNotEqualEqual_13count_nonzero_13/NotEqual/y*
_output_shapes	
:*
T0

p
count_nonzero_13/ToInt64Castcount_nonzero_13/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

`
count_nonzero_13/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

count_nonzero_13/SumSumcount_nonzero_13/ToInt64count_nonzero_13/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
T
ArgMax_9/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
f
ArgMax_9ArgMaxcond/Merge_1ArgMax_9/dimension*

Tidx0*
T0*
_output_shapes	
:
J
Equal_14EqualArgMax_9Const_5*
_output_shapes	
:*
T0	
]
count_nonzero_14/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
r
count_nonzero_14/NotEqualNotEqualEqual_14count_nonzero_14/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_14/ToInt64Castcount_nonzero_14/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
`
count_nonzero_14/ConstConst*
valueB: *
dtype0*
_output_shapes
:

count_nonzero_14/SumSumcount_nonzero_14/ToInt64count_nonzero_14/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
M
Greater_8/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
X
	Greater_8Greatercount_nonzero_13/SumGreater_8/y*
T0	*
_output_shapes
: 
P
cond_9/SwitchSwitch	Greater_8	Greater_8*
_output_shapes
: : *
T0

M
cond_9/switch_tIdentitycond_9/Switch:1*
T0
*
_output_shapes
: 
K
cond_9/switch_fIdentitycond_9/Switch*
_output_shapes
: *
T0

F
cond_9/pred_idIdentity	Greater_8*
T0
*
_output_shapes
: 

cond_9/truediv/Cast/SwitchSwitchcount_nonzero_12/Sumcond_9/pred_id*
T0	*
_output_shapes
: : *'
_class
loc:@count_nonzero_12/Sum
i
cond_9/truediv/CastCastcond_9/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	

cond_9/truediv/Cast_1/SwitchSwitchcount_nonzero_13/Sumcond_9/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_13/Sum*
T0	
m
cond_9/truediv/Cast_1Castcond_9/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
f
cond_9/truedivRealDivcond_9/truediv/Castcond_9/truediv/Cast_1*
T0*
_output_shapes
: 
g
cond_9/ConstConst^cond_9/switch_f*
valueB 2        *
_output_shapes
: *
dtype0
_
cond_9/MergeMergecond_9/Constcond_9/truediv*
_output_shapes
: : *
N*
T0
\
precision_4/tagsConst*
dtype0*
_output_shapes
: *
valueB Bprecision_4
]
precision_4ScalarSummaryprecision_4/tagscond_9/Merge*
T0*
_output_shapes
: 
M
Greater_9/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
X
	Greater_9Greatercount_nonzero_14/SumGreater_9/y*
T0	*
_output_shapes
: 
Q
cond_10/SwitchSwitch	Greater_9	Greater_9*
T0
*
_output_shapes
: : 
O
cond_10/switch_tIdentitycond_10/Switch:1*
T0
*
_output_shapes
: 
M
cond_10/switch_fIdentitycond_10/Switch*
T0
*
_output_shapes
: 
G
cond_10/pred_idIdentity	Greater_9*
_output_shapes
: *
T0


cond_10/truediv/Cast/SwitchSwitchcount_nonzero_12/Sumcond_10/pred_id*
T0	*
_output_shapes
: : *'
_class
loc:@count_nonzero_12/Sum
k
cond_10/truediv/CastCastcond_10/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	

cond_10/truediv/Cast_1/SwitchSwitchcount_nonzero_14/Sumcond_10/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_14/Sum*
T0	
o
cond_10/truediv/Cast_1Castcond_10/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
i
cond_10/truedivRealDivcond_10/truediv/Castcond_10/truediv/Cast_1*
T0*
_output_shapes
: 
i
cond_10/ConstConst^cond_10/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
b
cond_10/MergeMergecond_10/Constcond_10/truediv*
_output_shapes
: : *
T0*
N
V
recall_4/tagsConst*
_output_shapes
: *
dtype0*
valueB Brecall_4
X
recall_4ScalarSummaryrecall_4/tagscond_10/Merge*
_output_shapes
: *
T0
Ö
Const_6Const*
dtype0	*
_output_shapes	
:*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
U
ArgMax_10/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
h
	ArgMax_10ArgMaxcond/Merge_1ArgMax_10/dimension*

Tidx0*
T0*
_output_shapes	
:
K
Equal_15Equal	ArgMax_10Const_6*
T0	*
_output_shapes	
:
Q
LogicalAnd_5
LogicalAndaccuracy/EqualEqual_15*
_output_shapes	
:
]
count_nonzero_15/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
v
count_nonzero_15/NotEqualNotEqualLogicalAnd_5count_nonzero_15/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_15/ToInt64Castcount_nonzero_15/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

`
count_nonzero_15/ConstConst*
valueB: *
_output_shapes
:*
dtype0

count_nonzero_15/SumSumcount_nonzero_15/ToInt64count_nonzero_15/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
Z
Equal_16Equaloutput_projection/ArgMaxConst_6*
T0	*
_output_shapes	
:
]
count_nonzero_16/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

r
count_nonzero_16/NotEqualNotEqualEqual_16count_nonzero_16/NotEqual/y*
_output_shapes	
:*
T0

p
count_nonzero_16/ToInt64Castcount_nonzero_16/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
`
count_nonzero_16/ConstConst*
valueB: *
_output_shapes
:*
dtype0

count_nonzero_16/SumSumcount_nonzero_16/ToInt64count_nonzero_16/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
U
ArgMax_11/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
h
	ArgMax_11ArgMaxcond/Merge_1ArgMax_11/dimension*

Tidx0*
T0*
_output_shapes	
:
K
Equal_17Equal	ArgMax_11Const_6*
_output_shapes	
:*
T0	
]
count_nonzero_17/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
r
count_nonzero_17/NotEqualNotEqualEqual_17count_nonzero_17/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_17/ToInt64Castcount_nonzero_17/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

`
count_nonzero_17/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

count_nonzero_17/SumSumcount_nonzero_17/ToInt64count_nonzero_17/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
N
Greater_10/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
Z

Greater_10Greatercount_nonzero_16/SumGreater_10/y*
_output_shapes
: *
T0	
S
cond_11/SwitchSwitch
Greater_10
Greater_10*
_output_shapes
: : *
T0

O
cond_11/switch_tIdentitycond_11/Switch:1*
T0
*
_output_shapes
: 
M
cond_11/switch_fIdentitycond_11/Switch*
T0
*
_output_shapes
: 
H
cond_11/pred_idIdentity
Greater_10*
_output_shapes
: *
T0


cond_11/truediv/Cast/SwitchSwitchcount_nonzero_15/Sumcond_11/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_15/Sum*
T0	
k
cond_11/truediv/CastCastcond_11/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	

cond_11/truediv/Cast_1/SwitchSwitchcount_nonzero_16/Sumcond_11/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_16/Sum*
T0	
o
cond_11/truediv/Cast_1Castcond_11/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
i
cond_11/truedivRealDivcond_11/truediv/Castcond_11/truediv/Cast_1*
T0*
_output_shapes
: 
i
cond_11/ConstConst^cond_11/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
b
cond_11/MergeMergecond_11/Constcond_11/truediv*
_output_shapes
: : *
T0*
N
\
precision_5/tagsConst*
valueB Bprecision_5*
dtype0*
_output_shapes
: 
^
precision_5ScalarSummaryprecision_5/tagscond_11/Merge*
_output_shapes
: *
T0
N
Greater_11/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Z

Greater_11Greatercount_nonzero_17/SumGreater_11/y*
T0	*
_output_shapes
: 
S
cond_12/SwitchSwitch
Greater_11
Greater_11*
T0
*
_output_shapes
: : 
O
cond_12/switch_tIdentitycond_12/Switch:1*
_output_shapes
: *
T0

M
cond_12/switch_fIdentitycond_12/Switch*
T0
*
_output_shapes
: 
H
cond_12/pred_idIdentity
Greater_11*
T0
*
_output_shapes
: 

cond_12/truediv/Cast/SwitchSwitchcount_nonzero_15/Sumcond_12/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_15/Sum*
T0	
k
cond_12/truediv/CastCastcond_12/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	

cond_12/truediv/Cast_1/SwitchSwitchcount_nonzero_17/Sumcond_12/pred_id*
T0	*'
_class
loc:@count_nonzero_17/Sum*
_output_shapes
: : 
o
cond_12/truediv/Cast_1Castcond_12/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
i
cond_12/truedivRealDivcond_12/truediv/Castcond_12/truediv/Cast_1*
_output_shapes
: *
T0
i
cond_12/ConstConst^cond_12/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
b
cond_12/MergeMergecond_12/Constcond_12/truediv*
N*
T0*
_output_shapes
: : 
V
recall_5/tagsConst*
dtype0*
_output_shapes
: *
valueB Brecall_5
X
recall_5ScalarSummaryrecall_5/tagscond_12/Merge*
_output_shapes
: *
T0
Ö
Const_7Const*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                *
_output_shapes	
:*
dtype0	
U
ArgMax_12/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
h
	ArgMax_12ArgMaxcond/Merge_1ArgMax_12/dimension*

Tidx0*
T0*
_output_shapes	
:
K
Equal_18Equal	ArgMax_12Const_7*
_output_shapes	
:*
T0	
Q
LogicalAnd_6
LogicalAndaccuracy/EqualEqual_18*
_output_shapes	
:
]
count_nonzero_18/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

v
count_nonzero_18/NotEqualNotEqualLogicalAnd_6count_nonzero_18/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_18/ToInt64Castcount_nonzero_18/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

`
count_nonzero_18/ConstConst*
valueB: *
_output_shapes
:*
dtype0

count_nonzero_18/SumSumcount_nonzero_18/ToInt64count_nonzero_18/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
Z
Equal_19Equaloutput_projection/ArgMaxConst_7*
T0	*
_output_shapes	
:
]
count_nonzero_19/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
r
count_nonzero_19/NotEqualNotEqualEqual_19count_nonzero_19/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_19/ToInt64Castcount_nonzero_19/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
`
count_nonzero_19/ConstConst*
valueB: *
dtype0*
_output_shapes
:

count_nonzero_19/SumSumcount_nonzero_19/ToInt64count_nonzero_19/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
U
ArgMax_13/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
h
	ArgMax_13ArgMaxcond/Merge_1ArgMax_13/dimension*
_output_shapes	
:*
T0*

Tidx0
K
Equal_20Equal	ArgMax_13Const_7*
T0	*
_output_shapes	
:
]
count_nonzero_20/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
r
count_nonzero_20/NotEqualNotEqualEqual_20count_nonzero_20/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_20/ToInt64Castcount_nonzero_20/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

`
count_nonzero_20/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

count_nonzero_20/SumSumcount_nonzero_20/ToInt64count_nonzero_20/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
N
Greater_12/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Z

Greater_12Greatercount_nonzero_19/SumGreater_12/y*
_output_shapes
: *
T0	
S
cond_13/SwitchSwitch
Greater_12
Greater_12*
_output_shapes
: : *
T0

O
cond_13/switch_tIdentitycond_13/Switch:1*
T0
*
_output_shapes
: 
M
cond_13/switch_fIdentitycond_13/Switch*
T0
*
_output_shapes
: 
H
cond_13/pred_idIdentity
Greater_12*
_output_shapes
: *
T0


cond_13/truediv/Cast/SwitchSwitchcount_nonzero_18/Sumcond_13/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_18/Sum*
T0	
k
cond_13/truediv/CastCastcond_13/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	

cond_13/truediv/Cast_1/SwitchSwitchcount_nonzero_19/Sumcond_13/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_19/Sum*
T0	
o
cond_13/truediv/Cast_1Castcond_13/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
i
cond_13/truedivRealDivcond_13/truediv/Castcond_13/truediv/Cast_1*
_output_shapes
: *
T0
i
cond_13/ConstConst^cond_13/switch_f*
valueB 2        *
dtype0*
_output_shapes
: 
b
cond_13/MergeMergecond_13/Constcond_13/truediv*
_output_shapes
: : *
T0*
N
\
precision_6/tagsConst*
valueB Bprecision_6*
_output_shapes
: *
dtype0
^
precision_6ScalarSummaryprecision_6/tagscond_13/Merge*
_output_shapes
: *
T0
N
Greater_13/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
Z

Greater_13Greatercount_nonzero_20/SumGreater_13/y*
_output_shapes
: *
T0	
S
cond_14/SwitchSwitch
Greater_13
Greater_13*
T0
*
_output_shapes
: : 
O
cond_14/switch_tIdentitycond_14/Switch:1*
_output_shapes
: *
T0

M
cond_14/switch_fIdentitycond_14/Switch*
_output_shapes
: *
T0

H
cond_14/pred_idIdentity
Greater_13*
T0
*
_output_shapes
: 

cond_14/truediv/Cast/SwitchSwitchcount_nonzero_18/Sumcond_14/pred_id*
T0	*
_output_shapes
: : *'
_class
loc:@count_nonzero_18/Sum
k
cond_14/truediv/CastCastcond_14/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_14/truediv/Cast_1/SwitchSwitchcount_nonzero_20/Sumcond_14/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_20/Sum*
T0	
o
cond_14/truediv/Cast_1Castcond_14/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
i
cond_14/truedivRealDivcond_14/truediv/Castcond_14/truediv/Cast_1*
T0*
_output_shapes
: 
i
cond_14/ConstConst^cond_14/switch_f*
dtype0*
_output_shapes
: *
valueB 2        
b
cond_14/MergeMergecond_14/Constcond_14/truediv*
T0*
N*
_output_shapes
: : 
V
recall_6/tagsConst*
_output_shapes
: *
dtype0*
valueB Brecall_6
X
recall_6ScalarSummaryrecall_6/tagscond_14/Merge*
_output_shapes
: *
T0
Z
total_loss/tagsConst*
valueB B
total_loss*
_output_shapes
: *
dtype0
W

total_lossScalarSummarytotal_loss/tagsloss/Sum*
T0*
_output_shapes
: 
J
lr/tagsConst*
dtype0*
_output_shapes
: *
value
B Blr
L
lrScalarSummarylr/tagslearning_rate*
T0*
_output_shapes
: 
Z
accuracy_1/tagsConst*
valueB B
accuracy_1*
_output_shapes
: *
dtype0
`

accuracy_1ScalarSummaryaccuracy_1/tagsaccuracy/accuracy*
_output_shapes
: *
T0

Merge/MergeSummaryMergeSummaryprecision_0recall_0precision_1recall_1precision_2recall_2precision_3recall_3precision_4recall_4precision_5recall_5precision_6recall_6
total_losslr
accuracy_1*
N*
_output_shapes
: 
^
total_loss_1/tagsConst*
valueB Btotal_loss_1*
dtype0*
_output_shapes
: 
[
total_loss_1ScalarSummarytotal_loss_1/tagsloss/Sum*
T0*
_output_shapes
: 
N
	lr_1/tagsConst*
_output_shapes
: *
dtype0*
valueB
 Blr_1
P
lr_1ScalarSummary	lr_1/tagslearning_rate*
_output_shapes
: *
T0
Y
Merge_1/MergeSummaryMergeSummarytotal_loss_1lr_1*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Ń

save/SaveV2/tensor_namesConst*
_output_shapes
:$*
dtype0*

valueś	B÷	$BVariableBbeta1_powerBbeta2_powerBencoder/initial_state_0Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1Binput_embedBinput_embed/AdamBinput_embed/Adam_1Boutput_projection/WBoutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/bBoutput_projection/b/AdamBoutput_projection/b/Adam_1
«
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:$*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ü

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariablebeta1_powerbeta2_powerencoder/initial_state_0encoder/initial_state_0/Adamencoder/initial_state_0/Adam_1encoder/initial_state_1encoder/initial_state_1/Adamencoder/initial_state_1/Adam_1encoder/initial_state_2encoder/initial_state_2/Adamencoder/initial_state_2/Adam_1encoder/initial_state_3encoder/initial_state_3/Adamencoder/initial_state_3/Adam_12encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_13encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_12encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_13encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1input_embedinput_embed/Adaminput_embed/Adam_1output_projection/Woutput_projection/W/Adamoutput_projection/W/Adam_1output_projection/boutput_projection/b/Adamoutput_projection/b/Adam_1*2
dtypes(
&2$
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
_output_shapes
: *
T0
l
save/RestoreV2/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBBVariable
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:

save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
q
save/RestoreV2_1/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
 
save/Assign_1Assignbeta1_powersave/RestoreV2_1*
_class
loc:@input_embed*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
q
save/RestoreV2_2/tensor_namesConst*
dtype0*
_output_shapes
:* 
valueBBbeta2_power
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
 
save/Assign_2Assignbeta2_powersave/RestoreV2_2*
_class
loc:@input_embed*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
}
save/RestoreV2_3/tensor_namesConst*
dtype0*
_output_shapes
:*,
value#B!Bencoder/initial_state_0
j
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
Į
save/Assign_3Assignencoder/initial_state_0save/RestoreV2_3**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

save/RestoreV2_4/tensor_namesConst*
_output_shapes
:*
dtype0*1
value(B&Bencoder/initial_state_0/Adam
j
!save/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save/Assign_4Assignencoder/initial_state_0/Adamsave/RestoreV2_4*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(

save/RestoreV2_5/tensor_namesConst*3
value*B(Bencoder/initial_state_0/Adam_1*
_output_shapes
:*
dtype0
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
Č
save/Assign_5Assignencoder/initial_state_0/Adam_1save/RestoreV2_5*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0
}
save/RestoreV2_6/tensor_namesConst*
dtype0*
_output_shapes
:*,
value#B!Bencoder/initial_state_1
j
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
Į
save/Assign_6Assignencoder/initial_state_1save/RestoreV2_6*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_1*
T0*
use_locking(

save/RestoreV2_7/tensor_namesConst*1
value(B&Bencoder/initial_state_1/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
Ę
save/Assign_7Assignencoder/initial_state_1/Adamsave/RestoreV2_7*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	

save/RestoreV2_8/tensor_namesConst*3
value*B(Bencoder/initial_state_1/Adam_1*
_output_shapes
:*
dtype0
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
Č
save/Assign_8Assignencoder/initial_state_1/Adam_1save/RestoreV2_8*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_1*
T0*
use_locking(
}
save/RestoreV2_9/tensor_namesConst*,
value#B!Bencoder/initial_state_2*
_output_shapes
:*
dtype0
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Į
save/Assign_9Assignencoder/initial_state_2save/RestoreV2_9*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_2*
validate_shape(*
_output_shapes
:	

save/RestoreV2_10/tensor_namesConst*1
value(B&Bencoder/initial_state_2/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_10/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
Č
save/Assign_10Assignencoder/initial_state_2/Adamsave/RestoreV2_10**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

save/RestoreV2_11/tensor_namesConst*3
value*B(Bencoder/initial_state_2/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save/Assign_11Assignencoder/initial_state_2/Adam_1save/RestoreV2_11*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_2*
T0*
use_locking(
~
save/RestoreV2_12/tensor_namesConst*,
value#B!Bencoder/initial_state_3*
_output_shapes
:*
dtype0
k
"save/RestoreV2_12/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ć
save/Assign_12Assignencoder/initial_state_3save/RestoreV2_12*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_3*
T0*
use_locking(

save/RestoreV2_13/tensor_namesConst*1
value(B&Bencoder/initial_state_3/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
Č
save/Assign_13Assignencoder/initial_state_3/Adamsave/RestoreV2_13*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3

save/RestoreV2_14/tensor_namesConst*
dtype0*
_output_shapes
:*3
value*B(Bencoder/initial_state_3/Adam_1
k
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
_output_shapes
:*
dtypes
2
Ź
save/Assign_14Assignencoder/initial_state_3/Adam_1save/RestoreV2_14*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_3*
T0*
use_locking(

save/RestoreV2_15/tensor_namesConst*
dtype0*
_output_shapes
:*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
k
"save/RestoreV2_15/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
õ
save/Assign_15Assign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasessave/RestoreV2_15*
_output_shapes	
:*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
T0*
use_locking(

save/RestoreV2_16/tensor_namesConst*
_output_shapes
:*
dtype0*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam
k
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save/Assign_16Assign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adamsave/RestoreV2_16*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
validate_shape(*
_output_shapes	
:
 
save/RestoreV2_17/tensor_namesConst*
dtype0*
_output_shapes
:*N
valueEBCB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1
k
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
ü
save/Assign_17Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1save/RestoreV2_17*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(

save/RestoreV2_18/tensor_namesConst*H
value?B=B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
_output_shapes
:*
dtype0
k
"save/RestoreV2_18/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
_output_shapes
:*
dtypes
2
ü
save/Assign_18Assign3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightssave/RestoreV2_18* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0*
use_locking(

save/RestoreV2_19/tensor_namesConst*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_19/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_19Assign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamsave/RestoreV2_19*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
validate_shape(* 
_output_shapes
:

”
save/RestoreV2_20/tensor_namesConst*
dtype0*
_output_shapes
:*O
valueFBDB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1
k
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_20Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1save/RestoreV2_20* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0*
use_locking(

save/RestoreV2_21/tensor_namesConst*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes
:*
dtype0
k
"save/RestoreV2_21/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
õ
save/Assign_21Assign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasessave/RestoreV2_21*
_output_shapes	
:*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
use_locking(

save/RestoreV2_22/tensor_namesConst*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save/Assign_22Assign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adamsave/RestoreV2_22*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
 
save/RestoreV2_23/tensor_namesConst*N
valueEBCB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
ü
save/Assign_23Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1save/RestoreV2_23*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(

save/RestoreV2_24/tensor_namesConst*
dtype0*
_output_shapes
:*H
value?B=B3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
k
"save/RestoreV2_24/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
ü
save/Assign_24Assign3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightssave/RestoreV2_24*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights

save/RestoreV2_25/tensor_namesConst*
_output_shapes
:*
dtype0*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam
k
"save/RestoreV2_25/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_25Assign8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adamsave/RestoreV2_25* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0*
use_locking(
”
save/RestoreV2_26/tensor_namesConst*O
valueFBDB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_26/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_26Assign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1save/RestoreV2_26* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0*
use_locking(
r
save/RestoreV2_27/tensor_namesConst*
dtype0*
_output_shapes
:* 
valueBBinput_embed
k
"save/RestoreV2_27/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
Æ
save/Assign_27Assigninput_embedsave/RestoreV2_27*
_class
loc:@input_embed*#
_output_shapes
:*
T0*
validate_shape(*
use_locking(
w
save/RestoreV2_28/tensor_namesConst*%
valueBBinput_embed/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_28/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
_output_shapes
:*
dtypes
2
“
save/Assign_28Assigninput_embed/Adamsave/RestoreV2_28*#
_output_shapes
:*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
y
save/RestoreV2_29/tensor_namesConst*'
valueBBinput_embed/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_29/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
_output_shapes
:*
dtypes
2
¶
save/Assign_29Assigninput_embed/Adam_1save/RestoreV2_29*
use_locking(*
T0*
_class
loc:@input_embed*
validate_shape(*#
_output_shapes
:
z
save/RestoreV2_30/tensor_namesConst*(
valueBBoutput_projection/W*
_output_shapes
:*
dtype0
k
"save/RestoreV2_30/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
_output_shapes
:*
dtypes
2
»
save/Assign_30Assignoutput_projection/Wsave/RestoreV2_30*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

save/RestoreV2_31/tensor_namesConst*-
value$B"Boutput_projection/W/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_31/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
Ą
save/Assign_31Assignoutput_projection/W/Adamsave/RestoreV2_31*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*&
_class
loc:@output_projection/W

save/RestoreV2_32/tensor_namesConst*
_output_shapes
:*
dtype0*/
value&B$Boutput_projection/W/Adam_1
k
"save/RestoreV2_32/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
_output_shapes
:*
dtypes
2
Ā
save/Assign_32Assignoutput_projection/W/Adam_1save/RestoreV2_32*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
z
save/RestoreV2_33/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBoutput_projection/b
k
"save/RestoreV2_33/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save/Assign_33Assignoutput_projection/bsave/RestoreV2_33*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(

save/RestoreV2_34/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Boutput_projection/b/Adam
k
"save/RestoreV2_34/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
»
save/Assign_34Assignoutput_projection/b/Adamsave/RestoreV2_34*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(

save/RestoreV2_35/tensor_namesConst*
_output_shapes
:*
dtype0*/
value&B$Boutput_projection/b/Adam_1
k
"save/RestoreV2_35/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
½
save/Assign_35Assignoutput_projection/b/Adam_1save/RestoreV2_35*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(
š
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35

4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedVariable*
_output_shapes
: *
dtype0*
_class
loc:@Variable
”
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedinput_embed*
_class
loc:@input_embed*
_output_shapes
: *
dtype0
¹
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedencoder/initial_state_0**
_class 
loc:@encoder/initial_state_0*
dtype0*
_output_shapes
: 
¹
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedencoder/initial_state_1*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_1
¹
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedencoder/initial_state_2*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_2
¹
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedencoder/initial_state_3**
_class 
loc:@encoder/initial_state_3*
dtype0*
_output_shapes
: 
ń
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
dtype0*
_output_shapes
: 
ļ
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
dtype0*
_output_shapes
: *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
ń
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
_output_shapes
: *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
ļ
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitialized2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
²
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedoutput_projection/W*
_output_shapes
: *
dtype0*&
_class
loc:@output_projection/W
²
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedoutput_projection/b*&
_class
loc:@output_projection/b*
_output_shapes
: *
dtype0
¢
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedbeta1_power*
dtype0*
_output_shapes
: *
_class
loc:@input_embed
¢
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedbeta2_power*
_output_shapes
: *
dtype0*
_class
loc:@input_embed
§
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedinput_embed/Adam*
_output_shapes
: *
dtype0*
_class
loc:@input_embed
©
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedinput_embed/Adam_1*
_class
loc:@input_embed*
dtype0*
_output_shapes
: 
æ
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedencoder/initial_state_0/Adam*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_0
Į
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitializedencoder/initial_state_0/Adam_1*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_0
æ
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitializedencoder/initial_state_1/Adam**
_class 
loc:@encoder/initial_state_1*
_output_shapes
: *
dtype0
Į
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitializedencoder/initial_state_1/Adam_1**
_class 
loc:@encoder/initial_state_1*
_output_shapes
: *
dtype0
æ
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitializedencoder/initial_state_2/Adam*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_2
Į
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitializedencoder/initial_state_2/Adam_1*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_2
æ
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitializedencoder/initial_state_3/Adam**
_class 
loc:@encoder/initial_state_3*
_output_shapes
: *
dtype0
Į
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitializedencoder/initial_state_3/Adam_1**
_class 
loc:@encoder/initial_state_3*
dtype0*
_output_shapes
: 
÷
7report_uninitialized_variables/IsVariableInitialized_24IsVariableInitialized8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
_output_shapes
: *
dtype0
ł
7report_uninitialized_variables/IsVariableInitialized_25IsVariableInitialized:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
õ
7report_uninitialized_variables/IsVariableInitialized_26IsVariableInitialized7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*
dtype0*
_output_shapes
: *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
÷
7report_uninitialized_variables/IsVariableInitialized_27IsVariableInitialized9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
dtype0*
_output_shapes
: 
÷
7report_uninitialized_variables/IsVariableInitialized_28IsVariableInitialized8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
_output_shapes
: 
ł
7report_uninitialized_variables/IsVariableInitialized_29IsVariableInitialized:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
_output_shapes
: *
dtype0
õ
7report_uninitialized_variables/IsVariableInitialized_30IsVariableInitialized7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*
dtype0*
_output_shapes
: *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
÷
7report_uninitialized_variables/IsVariableInitialized_31IsVariableInitialized9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes
: *
dtype0
·
7report_uninitialized_variables/IsVariableInitialized_32IsVariableInitializedoutput_projection/W/Adam*&
_class
loc:@output_projection/W*
_output_shapes
: *
dtype0
¹
7report_uninitialized_variables/IsVariableInitialized_33IsVariableInitializedoutput_projection/W/Adam_1*&
_class
loc:@output_projection/W*
_output_shapes
: *
dtype0
·
7report_uninitialized_variables/IsVariableInitialized_34IsVariableInitializedoutput_projection/b/Adam*&
_class
loc:@output_projection/b*
_output_shapes
: *
dtype0
¹
7report_uninitialized_variables/IsVariableInitialized_35IsVariableInitializedoutput_projection/b/Adam_1*&
_class
loc:@output_projection/b*
_output_shapes
: *
dtype0
Ž
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_217report_uninitialized_variables/IsVariableInitialized_227report_uninitialized_variables/IsVariableInitialized_237report_uninitialized_variables/IsVariableInitialized_247report_uninitialized_variables/IsVariableInitialized_257report_uninitialized_variables/IsVariableInitialized_267report_uninitialized_variables/IsVariableInitialized_277report_uninitialized_variables/IsVariableInitialized_287report_uninitialized_variables/IsVariableInitialized_297report_uninitialized_variables/IsVariableInitialized_307report_uninitialized_variables/IsVariableInitialized_317report_uninitialized_variables/IsVariableInitialized_327report_uninitialized_variables/IsVariableInitialized_337report_uninitialized_variables/IsVariableInitialized_347report_uninitialized_variables/IsVariableInitialized_35*
N$*
T0
*
_output_shapes
:$*

axis 
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:$
Ż

$report_uninitialized_variables/ConstConst*
_output_shapes
:$*
dtype0*

valueś	B÷	$BVariableBinput_embedBencoder/initial_state_0Bencoder/initial_state_1Bencoder/initial_state_2Bencoder/initial_state_3B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesBoutput_projection/WBoutput_projection/bBbeta1_powerBbeta2_powerBinput_embed/AdamBinput_embed/Adam_1Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1B7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1Boutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/b/AdamBoutput_projection/b/Adam_1
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
_output_shapes
:*
dtype0*
valueB:$

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ł
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
end_mask *

begin_mask*
ellipsis_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
Index0*
T0

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
õ
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
valueB:$*
_output_shapes
:*
dtype0

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
valueB:*
_output_shapes
:*
dtype0

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
į
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
end_mask*
ellipsis_mask *

begin_mask *
shrink_axis_mask *
_output_shapes
: *
new_axis_mask *
T0*
Index0
Æ
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
T0*

axis *
N*
_output_shapes
:
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
«
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Ė
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
T0*
Tshape0*
_output_shapes
:$

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB:
’’’’’’’’’
Ū
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
T0
*
Tshape0*
_output_shapes
:$

1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:’’’’’’’’’
¶
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
T0	*#
_output_shapes
:’’’’’’’’’*
squeeze_dims


2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*#
_output_shapes
:’’’’’’’’’*
validate_indices(*
Tparams0*
Tindices0	

initNoOp^Variable/Assign^input_embed/Assign^encoder/initial_state_0/Assign^encoder/initial_state_1/Assign^encoder/initial_state_2/Assign^encoder/initial_state_3/Assign;^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Assign:^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Assign;^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Assign:^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Assign^output_projection/W/Assign^output_projection/b/Assign^beta1_power/Assign^beta2_power/Assign^input_embed/Adam/Assign^input_embed/Adam_1/Assign$^encoder/initial_state_0/Adam/Assign&^encoder/initial_state_0/Adam_1/Assign$^encoder/initial_state_1/Adam/Assign&^encoder/initial_state_1/Adam_1/Assign$^encoder/initial_state_2/Adam/Assign&^encoder/initial_state_2/Adam_1/Assign$^encoder/initial_state_3/Adam/Assign&^encoder/initial_state_3/Adam_1/Assign@^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/AssignB^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/Assign?^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/AssignA^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/Assign@^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/AssignB^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/Assign?^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/AssignA^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/Assign ^output_projection/W/Adam/Assign"^output_projection/W/Adam_1/Assign ^output_projection/b/Adam/Assign"^output_projection/b/Adam_1/Assign

init_1NoOp

init_all_tablesNoOp
-

group_depsNoOp^init_1^init_all_tables"58½     4’yŲ	!CÖAJū
ŻL¹L
+
Abs
x"T
y"T"
Ttype:	
2	
9
Add
x"T
y"T
z"T"
Ttype:
2	
S
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
Ń
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T"
Ttype:
2	"
use_lockingbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
p
	AssignAdd
ref"T

value"T

output_ref"T"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

ControlTrigger
É
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
ļ
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
ī
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

A
Equal
x"T
y"T
z
"
Ttype:
2	

)
Exit	
data"T
output"T"	
Ttype
+
Exp
x"T
y"T"
Ttype:	
2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
+
Floor
x"T
y"T"
Ttype:
2
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	

Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
:
Greater
x"T
y"T
z
"
Ttype:
2		
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
N
IsVariableInitialized
ref"dtype
is_initialized
"
dtypetype
<
L2Loss
t"T
output"T"
Ttype:
2	
7
Less
x"T
y"T
z
"
Ttype:
2		
-
Log1p
x"T
y"T"
Ttype:	
2
$

LogicalAnd
x

y

z



LogicalNot
x

y

!
LoopCond	
input


output

o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0

Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
:
Minimum
x"T
y"T
z"T"
Ttype:	
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	
-
Neg
x"T
y"T"
Ttype:
	2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
D
NotEqual
x"T
y"T
z
"
Ttype:
2	

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
µ
PaddingFIFOQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint’’’’’’’’’"
	containerstring "
shared_namestring 
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
5
Pow
x"T
y"T
z"T"
Ttype:
	2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
?
QueueCloseV2

handle"#
cancel_pending_enqueuesbool( 

QueueDequeueManyV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint’’’’’’’’’
~
QueueDequeueV2

handle

components2component_types"!
component_types
list(type)(0"

timeout_msint’’’’’’’’’
v
QueueEnqueueV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint’’’’’’’’’
#
QueueSizeV2

handle
size
ų
RandomShuffleQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint’’’’’’’’’"
min_after_dequeueint "
seedint "
seed2int "
	containerstring "
shared_namestring 
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
)
Rank

input"T

output"	
Ttype
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
4

Reciprocal
x"T
y"T"
Ttype:
	2	

RefEnter
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
/
Sigmoid
x"T
y"T"
Ttype:	
2
;
SigmoidGrad
x"T
y"T
z"T"
Ttype:	
2
.
Sign
x"T
y"T"
Ttype:
	2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
,
Sqrt
x"T
y"T"
Ttype:	
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
F
Stack
handle"
	elem_typetype"

stack_namestring 
?
StackPop
handle
elem"	elem_type"
	elem_typetype
V
	StackPush
handle	
elem"T
output"T"	
Ttype"
swap_memorybool( 
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
,
Tanh
x"T
y"T"
Ttype:	
2
8
TanhGrad
x"T
y"T
z"T"
Ttype:	
2
x
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring
V
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
a
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
6
TensorArraySizeV3

handle
flow_in
size
ø
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("
tensor_array_namestring 
]
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 

Where	
input
	
index	
&
	ZerosLike
x"T
y"T"	
Ttype*1.0.12v1.0.0-65-g4763edf-dirty±
f
PlaceholderPlaceholder*0
_output_shapes
:’’’’’’’’’’’’’’’’’’*
shape: *
dtype0
[
Placeholder_1Placeholder*
shape: *
dtype0*#
_output_shapes
:’’’’’’’’’
X

early_stopPlaceholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape: 
ė
random_queue_testRandomShuffleQueueV2*#
shapes
:	::*
_output_shapes
: *
component_types
2*
shared_name *
	container *
seed2ļ*

seed{*
min_after_dequeueč*
capacityč


random_queue_test_enqueueQueueEnqueueV2random_queue_testPlaceholderPlaceholder_1
early_stop*
Tcomponents
2*

timeout_ms’’’’’’’’’
¢
random_queue_test_DequeueQueueDequeueV2random_queue_test*+
_output_shapes
:	::*
component_types
2*

timeout_ms’’’’’’’’’
U
batch_and_pad/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z
Ä
 batch_and_pad/padding_fifo_queuePaddingFIFOQueueV2*#
shapes
:	::*
shared_name *
capacityč
*
_output_shapes
: *
component_types
2*
	container 
p
batch_and_pad/cond/SwitchSwitchbatch_and_pad/Constbatch_and_pad/Const*
T0
*
_output_shapes
: : 
e
batch_and_pad/cond/switch_tIdentitybatch_and_pad/cond/Switch:1*
_output_shapes
: *
T0

c
batch_and_pad/cond/switch_fIdentitybatch_and_pad/cond/Switch*
_output_shapes
: *
T0

\
batch_and_pad/cond/pred_idIdentitybatch_and_pad/Const*
_output_shapes
: *
T0

Ō
4batch_and_pad/cond/padding_fifo_queue_enqueue/SwitchSwitch batch_and_pad/padding_fifo_queuebatch_and_pad/cond/pred_id*3
_class)
'%loc:@batch_and_pad/padding_fifo_queue*
_output_shapes
: : *
T0
Ś
6batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_1Switchrandom_queue_test_Dequeuebatch_and_pad/cond/pred_id*,
_class"
 loc:@random_queue_test_Dequeue**
_output_shapes
:	:	*
T0
Ņ
6batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_2Switchrandom_queue_test_Dequeue:1batch_and_pad/cond/pred_id*,
_class"
 loc:@random_queue_test_Dequeue* 
_output_shapes
::*
T0
Ņ
6batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_3Switchrandom_queue_test_Dequeue:2batch_and_pad/cond/pred_id*
T0* 
_output_shapes
::*,
_class"
 loc:@random_queue_test_Dequeue
Ų
-batch_and_pad/cond/padding_fifo_queue_enqueueQueueEnqueueV26batch_and_pad/cond/padding_fifo_queue_enqueue/Switch:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_1:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_2:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_3:1*
Tcomponents
2*

timeout_ms’’’’’’’’’
Ļ
%batch_and_pad/cond/control_dependencyIdentitybatch_and_pad/cond/switch_t.^batch_and_pad/cond/padding_fifo_queue_enqueue*
T0
*.
_class$
" loc:@batch_and_pad/cond/switch_t*
_output_shapes
: 
=
batch_and_pad/cond/NoOpNoOp^batch_and_pad/cond/switch_f
»
'batch_and_pad/cond/control_dependency_1Identitybatch_and_pad/cond/switch_f^batch_and_pad/cond/NoOp*
T0
*
_output_shapes
: *.
_class$
" loc:@batch_and_pad/cond/switch_f

batch_and_pad/cond/MergeMerge'batch_and_pad/cond/control_dependency_1%batch_and_pad/cond/control_dependency*
_output_shapes
: : *
N*
T0

w
&batch_and_pad/padding_fifo_queue_CloseQueueCloseV2 batch_and_pad/padding_fifo_queue*
cancel_pending_enqueues( 
y
(batch_and_pad/padding_fifo_queue_Close_1QueueCloseV2 batch_and_pad/padding_fifo_queue*
cancel_pending_enqueues(
n
%batch_and_pad/padding_fifo_queue_SizeQueueSizeV2 batch_and_pad/padding_fifo_queue*
_output_shapes
: 
q
batch_and_pad/CastCast%batch_and_pad/padding_fifo_queue_Size*
_output_shapes
: *

DstT0*

SrcT0
X
batch_and_pad/mul/yConst*
dtype0*
_output_shapes
: *
valueB
 *i=:
b
batch_and_pad/mulMulbatch_and_pad/Castbatch_and_pad/mul/y*
T0*
_output_shapes
: 

(batch_and_pad/fraction_of_1384_full/tagsConst*
dtype0*
_output_shapes
: *4
value+B) B#batch_and_pad/fraction_of_1384_full

#batch_and_pad/fraction_of_1384_fullScalarSummary(batch_and_pad/fraction_of_1384_full/tagsbatch_and_pad/mul*
T0*
_output_shapes
: 
R
batch_and_pad/nConst*
value
B :*
dtype0*
_output_shapes
: 
É
batch_and_padQueueDequeueManyV2 batch_and_pad/padding_fifo_queuebatch_and_pad/n*:
_output_shapes(
&::	:	*
component_types
2*

timeout_ms’’’’’’’’’
h
Placeholder_2Placeholder*
dtype0*
shape: *0
_output_shapes
:’’’’’’’’’’’’’’’’’’
[
Placeholder_3Placeholder*
shape: *
dtype0*#
_output_shapes
:’’’’’’’’’
Z
early_stop_1Placeholder*
shape: *
dtype0*#
_output_shapes
:’’’’’’’’’
ė
random_queue_trainRandomShuffleQueueV2*#
shapes
:	::*
	container *
seed2{*

seed{*
_output_shapes
: *
component_types
2*
min_after_dequeueč*
capacityč
*
shared_name 

random_queue_train_enqueueQueueEnqueueV2random_queue_trainPlaceholder_2Placeholder_3early_stop_1*
Tcomponents
2*

timeout_ms’’’’’’’’’
¤
random_queue_train_DequeueQueueDequeueV2random_queue_train*

timeout_ms’’’’’’’’’*+
_output_shapes
:	::*
component_types
2
W
batch_and_pad_1/ConstConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
Ę
"batch_and_pad_1/padding_fifo_queuePaddingFIFOQueueV2*#
shapes
:	::*
capacityč
*
	container *
shared_name *
_output_shapes
: *
component_types
2
v
batch_and_pad_1/cond/SwitchSwitchbatch_and_pad_1/Constbatch_and_pad_1/Const*
T0
*
_output_shapes
: : 
i
batch_and_pad_1/cond/switch_tIdentitybatch_and_pad_1/cond/Switch:1*
T0
*
_output_shapes
: 
g
batch_and_pad_1/cond/switch_fIdentitybatch_and_pad_1/cond/Switch*
_output_shapes
: *
T0

`
batch_and_pad_1/cond/pred_idIdentitybatch_and_pad_1/Const*
_output_shapes
: *
T0

Ü
6batch_and_pad_1/cond/padding_fifo_queue_enqueue/SwitchSwitch"batch_and_pad_1/padding_fifo_queuebatch_and_pad_1/cond/pred_id*
T0*
_output_shapes
: : *5
_class+
)'loc:@batch_and_pad_1/padding_fifo_queue
ą
8batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_1Switchrandom_queue_train_Dequeuebatch_and_pad_1/cond/pred_id*-
_class#
!loc:@random_queue_train_Dequeue**
_output_shapes
:	:	*
T0
Ų
8batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_2Switchrandom_queue_train_Dequeue:1batch_and_pad_1/cond/pred_id*-
_class#
!loc:@random_queue_train_Dequeue* 
_output_shapes
::*
T0
Ų
8batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_3Switchrandom_queue_train_Dequeue:2batch_and_pad_1/cond/pred_id*
T0* 
_output_shapes
::*-
_class#
!loc:@random_queue_train_Dequeue
ā
/batch_and_pad_1/cond/padding_fifo_queue_enqueueQueueEnqueueV28batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch:1:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_1:1:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_2:1:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_3:1*
Tcomponents
2*

timeout_ms’’’’’’’’’
×
'batch_and_pad_1/cond/control_dependencyIdentitybatch_and_pad_1/cond/switch_t0^batch_and_pad_1/cond/padding_fifo_queue_enqueue*
T0
*0
_class&
$"loc:@batch_and_pad_1/cond/switch_t*
_output_shapes
: 
A
batch_and_pad_1/cond/NoOpNoOp^batch_and_pad_1/cond/switch_f
Ć
)batch_and_pad_1/cond/control_dependency_1Identitybatch_and_pad_1/cond/switch_f^batch_and_pad_1/cond/NoOp*
_output_shapes
: *0
_class&
$"loc:@batch_and_pad_1/cond/switch_f*
T0

£
batch_and_pad_1/cond/MergeMerge)batch_and_pad_1/cond/control_dependency_1'batch_and_pad_1/cond/control_dependency*
T0
*
N*
_output_shapes
: : 
{
(batch_and_pad_1/padding_fifo_queue_CloseQueueCloseV2"batch_and_pad_1/padding_fifo_queue*
cancel_pending_enqueues( 
}
*batch_and_pad_1/padding_fifo_queue_Close_1QueueCloseV2"batch_and_pad_1/padding_fifo_queue*
cancel_pending_enqueues(
r
'batch_and_pad_1/padding_fifo_queue_SizeQueueSizeV2"batch_and_pad_1/padding_fifo_queue*
_output_shapes
: 
u
batch_and_pad_1/CastCast'batch_and_pad_1/padding_fifo_queue_Size*
_output_shapes
: *

DstT0*

SrcT0
Z
batch_and_pad_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *i=:
h
batch_and_pad_1/mulMulbatch_and_pad_1/Castbatch_and_pad_1/mul/y*
_output_shapes
: *
T0

*batch_and_pad_1/fraction_of_1384_full/tagsConst*6
value-B+ B%batch_and_pad_1/fraction_of_1384_full*
_output_shapes
: *
dtype0

%batch_and_pad_1/fraction_of_1384_fullScalarSummary*batch_and_pad_1/fraction_of_1384_full/tagsbatch_and_pad_1/mul*
_output_shapes
: *
T0
T
batch_and_pad_1/nConst*
_output_shapes
: *
dtype0*
value
B :
Ļ
batch_and_pad_1QueueDequeueManyV2"batch_and_pad_1/padding_fifo_queuebatch_and_pad_1/n*

timeout_ms’’’’’’’’’*:
_output_shapes(
&::	:	*
component_types
2
G
ConstConst*
value	B
 Z *
_output_shapes
: *
dtype0

^
is_trainingPlaceholderWithDefaultConst*
dtype0
*
shape: *
_output_shapes
: 
R
cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

I
cond/switch_tIdentitycond/Switch:1*
_output_shapes
: *
T0

G
cond/switch_fIdentitycond/Switch*
_output_shapes
: *
T0

F
cond/pred_idIdentityis_training*
_output_shapes
: *
T0


cond/Switch_1Switchbatch_and_pad_1cond/pred_id*
T0*"
_class
loc:@batch_and_pad_1*4
_output_shapes"
 ::

cond/Switch_2Switchbatch_and_pad_1:1cond/pred_id*
T0*"
_class
loc:@batch_and_pad_1**
_output_shapes
:	:	

cond/Switch_3Switchbatch_and_pad_1:2cond/pred_id**
_output_shapes
:	:	*"
_class
loc:@batch_and_pad_1*
T0

cond/Switch_4Switchbatch_and_padcond/pred_id*4
_output_shapes"
 ::* 
_class
loc:@batch_and_pad*
T0

cond/Switch_5Switchbatch_and_pad:1cond/pred_id**
_output_shapes
:	:	* 
_class
loc:@batch_and_pad*
T0

cond/Switch_6Switchbatch_and_pad:2cond/pred_id**
_output_shapes
:	:	* 
_class
loc:@batch_and_pad*
T0
m

cond/MergeMergecond/Switch_4cond/Switch_1:1*&
_output_shapes
:: *
N*
T0
j
cond/Merge_1Mergecond/Switch_5cond/Switch_2:1*!
_output_shapes
:	: *
T0*
N
j
cond/Merge_2Mergecond/Switch_6cond/Switch_3:1*
T0*
N*!
_output_shapes
:	: 
j
cond/Merge_3Mergecond/Switch_6cond/Switch_3:1*
T0*
N*!
_output_shapes
:	: 
`
Reshape/shapeConst*
valueB:
’’’’’’’’’*
_output_shapes
:*
dtype0
c
ReshapeReshapecond/Merge_2Reshape/shape*
Tshape0*
_output_shapes	
:*
T0
X
Variable/initial_valueConst*
value	B : *
_output_shapes
: *
dtype0
l
Variable
VariableV2*
shared_name *
dtype0*
shape: *
_output_shapes
: *
	container 
¢
Variable/AssignAssignVariableVariable/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
a
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes
: 
”
,input_embed/Initializer/random_uniform/shapeConst*
_class
loc:@input_embed*!
valueB"         *
dtype0*
_output_shapes
:

*input_embed/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
_class
loc:@input_embed*
valueB
 *
×£½

*input_embed/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
_class
loc:@input_embed*
valueB
 *
×£=
ē
4input_embed/Initializer/random_uniform/RandomUniformRandomUniform,input_embed/Initializer/random_uniform/shape*#
_output_shapes
:*
_class
loc:@input_embed*
dtype0*

seed{*
T0*
seed2W
Ź
*input_embed/Initializer/random_uniform/subSub*input_embed/Initializer/random_uniform/max*input_embed/Initializer/random_uniform/min*
_output_shapes
: *
_class
loc:@input_embed*
T0
į
*input_embed/Initializer/random_uniform/mulMul4input_embed/Initializer/random_uniform/RandomUniform*input_embed/Initializer/random_uniform/sub*
_class
loc:@input_embed*#
_output_shapes
:*
T0
Ó
&input_embed/Initializer/random_uniformAdd*input_embed/Initializer/random_uniform/mul*input_embed/Initializer/random_uniform/min*#
_output_shapes
:*
_class
loc:@input_embed*
T0
©
input_embed
VariableV2*
shape:*#
_output_shapes
:*
shared_name *
_class
loc:@input_embed*
dtype0*
	container 
Č
input_embed/AssignAssigninput_embed&input_embed/Initializer/random_uniform*#
_output_shapes
:*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
w
input_embed/readIdentityinput_embed*
T0*
_class
loc:@input_embed*#
_output_shapes
:
_
encoder/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

encoder/conv1d/ExpandDims
ExpandDims
cond/Mergeencoder/conv1d/ExpandDims/dim*

Tdim0*(
_output_shapes
:*
T0
a
encoder/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 

encoder/conv1d/ExpandDims_1
ExpandDimsinput_embed/readencoder/conv1d/ExpandDims_1/dim*'
_output_shapes
:*
T0*

Tdim0
ć
encoder/conv1d/Conv2DConv2Dencoder/conv1d/ExpandDimsencoder/conv1d/ExpandDims_1*)
_output_shapes
:*
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingVALID

encoder/conv1d/SqueezeSqueezeencoder/conv1d/Conv2D*%
_output_shapes
:*
T0*
squeeze_dims

Z
ShapeConst*
dtype0*
_output_shapes
:*!
valueB"         
]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
_
strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
ł
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask*
T0*
Index0*
end_mask *
_output_shapes
: *

begin_mask *
ellipsis_mask 
¬
)encoder/initial_state_0/Initializer/ConstConst*
dtype0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0*
valueB	*    
¹
encoder/initial_state_0
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
shape:	*
shared_name 
ė
encoder/initial_state_0/AssignAssignencoder/initial_state_0)encoder/initial_state_0/Initializer/Const*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(

encoder/initial_state_0/readIdentityencoder/initial_state_0**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
T0
m
+encoder_1/initial_state_0_tiled/multiples/1Const*
value	B :*
_output_shapes
: *
dtype0
§
)encoder_1/initial_state_0_tiled/multiplesPackstrided_slice+encoder_1/initial_state_0_tiled/multiples/1*

axis *
_output_shapes
:*
T0*
N
µ
encoder_1/initial_state_0_tiledTileencoder/initial_state_0/read)encoder_1/initial_state_0_tiled/multiples*

Tmultiples0*
T0*(
_output_shapes
:’’’’’’’’’
¬
)encoder/initial_state_1/Initializer/ConstConst*
dtype0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1*
valueB	*    
¹
encoder/initial_state_1
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	*
shape:	*
shared_name 
ė
encoder/initial_state_1/AssignAssignencoder/initial_state_1)encoder/initial_state_1/Initializer/Const**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

encoder/initial_state_1/readIdentityencoder/initial_state_1*
T0**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	
m
+encoder_1/initial_state_1_tiled/multiples/1Const*
value	B :*
_output_shapes
: *
dtype0
§
)encoder_1/initial_state_1_tiled/multiplesPackstrided_slice+encoder_1/initial_state_1_tiled/multiples/1*
N*
T0*
_output_shapes
:*

axis 
µ
encoder_1/initial_state_1_tiledTileencoder/initial_state_1/read)encoder_1/initial_state_1_tiled/multiples*(
_output_shapes
:’’’’’’’’’*
T0*

Tmultiples0
¬
)encoder/initial_state_2/Initializer/ConstConst*
_output_shapes
:	*
dtype0**
_class 
loc:@encoder/initial_state_2*
valueB	*    
¹
encoder/initial_state_2
VariableV2*
shared_name *
shape:	*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2*
dtype0*
	container 
ė
encoder/initial_state_2/AssignAssignencoder/initial_state_2)encoder/initial_state_2/Initializer/Const*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_2*
validate_shape(*
_output_shapes
:	

encoder/initial_state_2/readIdentityencoder/initial_state_2*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2
m
+encoder_1/initial_state_2_tiled/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
§
)encoder_1/initial_state_2_tiled/multiplesPackstrided_slice+encoder_1/initial_state_2_tiled/multiples/1*

axis *
_output_shapes
:*
T0*
N
µ
encoder_1/initial_state_2_tiledTileencoder/initial_state_2/read)encoder_1/initial_state_2_tiled/multiples*

Tmultiples0*
T0*(
_output_shapes
:’’’’’’’’’
¬
)encoder/initial_state_3/Initializer/ConstConst*
_output_shapes
:	*
dtype0**
_class 
loc:@encoder/initial_state_3*
valueB	*    
¹
encoder/initial_state_3
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_3*
shared_name *
_output_shapes
:	*
shape:	
ė
encoder/initial_state_3/AssignAssignencoder/initial_state_3)encoder/initial_state_3/Initializer/Const*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_3*
T0*
use_locking(

encoder/initial_state_3/readIdentityencoder/initial_state_3*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3*
T0
m
+encoder_1/initial_state_3_tiled/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
§
)encoder_1/initial_state_3_tiled/multiplesPackstrided_slice+encoder_1/initial_state_3_tiled/multiples/1*
_output_shapes
:*
N*

axis *
T0
µ
encoder_1/initial_state_3_tiledTileencoder/initial_state_3/read)encoder_1/initial_state_3_tiled/multiples*

Tmultiples0*
T0*(
_output_shapes
:’’’’’’’’’
m
encoder_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

encoder_1/transpose	Transposeencoder/conv1d/Squeezeencoder_1/transpose/perm*
Tperm0*
T0*%
_output_shapes
:
T
encoder_1/sequence_lengthIdentityReshape*
T0*
_output_shapes	
:
h
encoder_1/rnn/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
k
!encoder_1/rnn/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
m
#encoder_1/rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
m
#encoder_1/rnn/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
æ
encoder_1/rnn/strided_sliceStridedSliceencoder_1/rnn/Shape!encoder_1/rnn/strided_slice/stack#encoder_1/rnn/strided_slice/stack_1#encoder_1/rnn/strided_slice/stack_2*
end_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask*
_output_shapes
: *
new_axis_mask *
T0*
Index0
m
#encoder_1/rnn/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
o
%encoder_1/rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
o
%encoder_1/rnn/strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ē
encoder_1/rnn/strided_slice_1StridedSliceencoder_1/rnn/Shape#encoder_1/rnn/strided_slice_1/stack%encoder_1/rnn/strided_slice_1/stack_1%encoder_1/rnn/strided_slice_1/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0*
end_mask *
new_axis_mask *
ellipsis_mask *

begin_mask 
`
encoder_1/rnn/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
r
encoder_1/rnn/stackPackencoder_1/rnn/strided_slice*
_output_shapes
:*
N*

axis *
T0
m
encoder_1/rnn/EqualEqualencoder_1/rnn/Shape_1encoder_1/rnn/stack*
T0*
_output_shapes
:
]
encoder_1/rnn/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
w
encoder_1/rnn/AllAllencoder_1/rnn/Equalencoder_1/rnn/Const*

Tidx0*
	keep_dims( *
_output_shapes
: 

encoder_1/rnn/Assert/ConstConst*
_output_shapes
: *
dtype0*J
valueAB? B9Expected shape for Tensor encoder_1/sequence_length:0 is 
m
encoder_1/rnn/Assert/Const_1Const*
_output_shapes
: *
dtype0*!
valueB B but saw shape: 

"encoder_1/rnn/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*J
valueAB? B9Expected shape for Tensor encoder_1/sequence_length:0 is 
s
"encoder_1/rnn/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*!
valueB B but saw shape: 
Ģ
encoder_1/rnn/Assert/AssertAssertencoder_1/rnn/All"encoder_1/rnn/Assert/Assert/data_0encoder_1/rnn/stack"encoder_1/rnn/Assert/Assert/data_2encoder_1/rnn/Shape_1*
	summarize*
T
2

encoder_1/rnn/CheckSeqLenIdentityencoder_1/sequence_length^encoder_1/rnn/Assert/Assert*
T0*
_output_shapes	
:
j
encoder_1/rnn/Shape_2Const*
dtype0*
_output_shapes
:*!
valueB"         
m
#encoder_1/rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
o
%encoder_1/rnn/strided_slice_2/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
o
%encoder_1/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
É
encoder_1/rnn/strided_slice_2StridedSliceencoder_1/rnn/Shape_2#encoder_1/rnn/strided_slice_2/stack%encoder_1/rnn/strided_slice_2/stack_1%encoder_1/rnn/strided_slice_2/stack_2*
end_mask *
ellipsis_mask *

begin_mask *
shrink_axis_mask*
_output_shapes
: *
new_axis_mask *
T0*
Index0
m
#encoder_1/rnn/strided_slice_3/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%encoder_1/rnn/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
o
%encoder_1/rnn/strided_slice_3/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
É
encoder_1/rnn/strided_slice_3StridedSliceencoder_1/rnn/Shape_2#encoder_1/rnn/strided_slice_3/stack%encoder_1/rnn/strided_slice_3/stack_1%encoder_1/rnn/strided_slice_3/stack_2*
_output_shapes
: *
end_mask *
new_axis_mask *
ellipsis_mask *

begin_mask *
shrink_axis_mask*
T0*
Index0
Z
encoder_1/rnn/stack_1/1Const*
dtype0*
_output_shapes
: *
value
B :

encoder_1/rnn/stack_1Packencoder_1/rnn/strided_slice_3encoder_1/rnn/stack_1/1*
_output_shapes
:*
N*

axis *
T0
^
encoder_1/rnn/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0

encoder_1/rnn/zerosFillencoder_1/rnn/stack_1encoder_1/rnn/zeros/Const*
T0*(
_output_shapes
:’’’’’’’’’
_
encoder_1/rnn/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

encoder_1/rnn/MinMinencoder_1/rnn/CheckSeqLenencoder_1/rnn/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
_
encoder_1/rnn/Const_2Const*
valueB: *
dtype0*
_output_shapes
:

encoder_1/rnn/MaxMaxencoder_1/rnn/CheckSeqLenencoder_1/rnn/Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
T
encoder_1/rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 
ō
encoder_1/rnn/TensorArrayTensorArrayV3encoder_1/rnn/strided_slice_2*
element_shape:*
dynamic_size( *
clear_after_read(*9
tensor_array_name$"encoder_1/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

::
õ
encoder_1/rnn/TensorArray_1TensorArrayV3encoder_1/rnn/strided_slice_2*
element_shape:*
dynamic_size( *
clear_after_read(*8
tensor_array_name#!encoder_1/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

::
{
&encoder_1/rnn/TensorArrayUnstack/ShapeConst*
dtype0*
_output_shapes
:*!
valueB"         
~
4encoder_1/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0

.encoder_1/rnn/TensorArrayUnstack/strided_sliceStridedSlice&encoder_1/rnn/TensorArrayUnstack/Shape4encoder_1/rnn/TensorArrayUnstack/strided_slice/stack6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_16encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask*
T0*
Index0*
end_mask *
_output_shapes
: *

begin_mask *
ellipsis_mask 
n
,encoder_1/rnn/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
n
,encoder_1/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
ģ
&encoder_1/rnn/TensorArrayUnstack/rangeRange,encoder_1/rnn/TensorArrayUnstack/range/start.encoder_1/rnn/TensorArrayUnstack/strided_slice,encoder_1/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:’’’’’’’’’*

Tidx0
Ŗ
Hencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3encoder_1/rnn/TensorArray_1&encoder_1/rnn/TensorArrayUnstack/rangeencoder_1/transposeencoder_1/rnn/TensorArray_1:1*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: 
æ
encoder_1/rnn/while/EnterEnterencoder_1/rnn/time*
parallel_iterations *
T0*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
Ģ
encoder_1/rnn/while/Enter_1Enterencoder_1/rnn/TensorArray:1*
parallel_iterations *
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
ą
encoder_1/rnn/while/Enter_2Enterencoder_1/initial_state_0_tiled*(
_output_shapes
:’’’’’’’’’*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
ą
encoder_1/rnn/while/Enter_3Enterencoder_1/initial_state_1_tiled*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:’’’’’’’’’*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
ą
encoder_1/rnn/while/Enter_4Enterencoder_1/initial_state_2_tiled*
parallel_iterations *
T0*(
_output_shapes
:’’’’’’’’’*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
ą
encoder_1/rnn/while/Enter_5Enterencoder_1/initial_state_3_tiled*(
_output_shapes
:’’’’’’’’’*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0

encoder_1/rnn/while/MergeMergeencoder_1/rnn/while/Enter!encoder_1/rnn/while/NextIteration*
_output_shapes
: : *
N*
T0

encoder_1/rnn/while/Merge_1Mergeencoder_1/rnn/while/Enter_1#encoder_1/rnn/while/NextIteration_1*
_output_shapes
:: *
N*
T0
¤
encoder_1/rnn/while/Merge_2Mergeencoder_1/rnn/while/Enter_2#encoder_1/rnn/while/NextIteration_2*
T0*
N**
_output_shapes
:’’’’’’’’’: 
¤
encoder_1/rnn/while/Merge_3Mergeencoder_1/rnn/while/Enter_3#encoder_1/rnn/while/NextIteration_3**
_output_shapes
:’’’’’’’’’: *
T0*
N
¤
encoder_1/rnn/while/Merge_4Mergeencoder_1/rnn/while/Enter_4#encoder_1/rnn/while/NextIteration_4*
N*
T0**
_output_shapes
:’’’’’’’’’: 
¤
encoder_1/rnn/while/Merge_5Mergeencoder_1/rnn/while/Enter_5#encoder_1/rnn/while/NextIteration_5**
_output_shapes
:’’’’’’’’’: *
T0*
N
Ļ
encoder_1/rnn/while/Less/EnterEnterencoder_1/rnn/strided_slice_2*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
|
encoder_1/rnn/while/LessLessencoder_1/rnn/while/Mergeencoder_1/rnn/while/Less/Enter*
T0*
_output_shapes
: 
Z
encoder_1/rnn/while/LoopCondLoopCondencoder_1/rnn/while/Less*
_output_shapes
: 
®
encoder_1/rnn/while/SwitchSwitchencoder_1/rnn/while/Mergeencoder_1/rnn/while/LoopCond*,
_class"
 loc:@encoder_1/rnn/while/Merge*
_output_shapes
: : *
T0
ø
encoder_1/rnn/while/Switch_1Switchencoder_1/rnn/while/Merge_1encoder_1/rnn/while/LoopCond*
T0*
_output_shapes

::*.
_class$
" loc:@encoder_1/rnn/while/Merge_1
Ų
encoder_1/rnn/while/Switch_2Switchencoder_1/rnn/while/Merge_2encoder_1/rnn/while/LoopCond*
T0*<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*.
_class$
" loc:@encoder_1/rnn/while/Merge_2
Ų
encoder_1/rnn/while/Switch_3Switchencoder_1/rnn/while/Merge_3encoder_1/rnn/while/LoopCond*
T0*<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*.
_class$
" loc:@encoder_1/rnn/while/Merge_3
Ų
encoder_1/rnn/while/Switch_4Switchencoder_1/rnn/while/Merge_4encoder_1/rnn/while/LoopCond*<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*.
_class$
" loc:@encoder_1/rnn/while/Merge_4*
T0
Ų
encoder_1/rnn/while/Switch_5Switchencoder_1/rnn/while/Merge_5encoder_1/rnn/while/LoopCond*
T0*.
_class$
" loc:@encoder_1/rnn/while/Merge_5*<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’
g
encoder_1/rnn/while/IdentityIdentityencoder_1/rnn/while/Switch:1*
_output_shapes
: *
T0
m
encoder_1/rnn/while/Identity_1Identityencoder_1/rnn/while/Switch_1:1*
T0*
_output_shapes
:
}
encoder_1/rnn/while/Identity_2Identityencoder_1/rnn/while/Switch_2:1*
T0*(
_output_shapes
:’’’’’’’’’
}
encoder_1/rnn/while/Identity_3Identityencoder_1/rnn/while/Switch_3:1*(
_output_shapes
:’’’’’’’’’*
T0
}
encoder_1/rnn/while/Identity_4Identityencoder_1/rnn/while/Switch_4:1*(
_output_shapes
:’’’’’’’’’*
T0
}
encoder_1/rnn/while/Identity_5Identityencoder_1/rnn/while/Switch_5:1*
T0*(
_output_shapes
:’’’’’’’’’

+encoder_1/rnn/while/TensorArrayReadV3/EnterEnterencoder_1/rnn/TensorArray_1*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
¹
-encoder_1/rnn/while/TensorArrayReadV3/Enter_1EnterHencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 

%encoder_1/rnn/while/TensorArrayReadV3TensorArrayReadV3+encoder_1/rnn/while/TensorArrayReadV3/Enterencoder_1/rnn/while/Identity-encoder_1/rnn/while/TensorArrayReadV3/Enter_1*
dtype0* 
_output_shapes
:
*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
ķ
Tencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB"      
ß
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/minConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB
 *
×£½*
_output_shapes
: *
dtype0
ß
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB
 *
×£=
Ż
\encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/RandomUniformRandomUniformTencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/shape*
seed2Ś*
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*

seed{* 
_output_shapes
:
*
T0
ź
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/subSubRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/maxRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/min*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0
ž
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/mulMul\encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/RandomUniformRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/sub* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0
š
Nencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniformAddRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/mulRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/min*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0
ó
3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
VariableV2* 
_output_shapes
:
*
dtype0*
shape:
*
	container *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
shared_name 
å
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AssignAssign3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsNencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
¤
8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/readIdentity3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0
Ŗ
Iencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axisConst^encoder_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
¢
Dencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concatConcatV2%encoder_1/rnn/while/TensorArrayReadV3encoder_1/rnn/while/Identity_3Iencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
N*

Tidx0*
T0* 
_output_shapes
:

 
Jencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read*
is_constant(* 
_output_shapes
:
*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
±
Dencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMulMatMulDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concatJencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter*
transpose_b( * 
_output_shapes
:
*
transpose_a( *
T0
Ś
Dencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Initializer/ConstConst*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
valueB*    *
dtype0*
_output_shapes	
:
ē
2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
VariableV2*
	container *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
shared_name *
_output_shapes	
:*
shape:
Ó
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AssignAssign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesDencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Initializer/Const*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(

7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/readIdentity2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
T0*
_output_shapes	
:

Aencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/EnterEnter7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read*
parallel_iterations *
T0*
_output_shapes	
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(

;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAddBiasAddDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMulAencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC* 
_output_shapes
:

¤
Cencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dimConst^encoder_1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
¤
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/splitSplitCencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd*D
_output_shapes2
0:
:
:
:
*
	num_split*
T0

9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add/yConst^encoder_1/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  ?
į
7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/addAdd;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add/y* 
_output_shapes
:
*
T0
Ŗ
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/SigmoidSigmoid7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add*
T0* 
_output_shapes
:

Ę
7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mulMul;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoidencoder_1/rnn/while/Identity_2* 
_output_shapes
:
*
T0
®
=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1Sigmoid9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split*
T0* 
_output_shapes
:

Ø
8encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/TanhTanh;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:1* 
_output_shapes
:
*
T0
ä
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1Mul=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_18encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
T0* 
_output_shapes
:

ß
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1Add7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1* 
_output_shapes
:
*
T0
°
=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2Sigmoid;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:3* 
_output_shapes
:
*
T0
Ø
:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1Tanh9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1*
T0* 
_output_shapes
:

ę
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2Mul=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1* 
_output_shapes
:
*
T0
ķ
Tencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB"      
ß
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/minConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB
 *
×£½*
dtype0*
_output_shapes
: 
ß
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/maxConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB
 *
×£=*
_output_shapes
: *
dtype0
Ż
\encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/RandomUniformRandomUniformTencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/shape*

seed{*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
seed2ū*
dtype0* 
_output_shapes
:

ź
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/subSubRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/maxRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
ž
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/mulMul\encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/RandomUniformRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/sub* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0
š
Nencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniformAddRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/mulRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
ó
3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
VariableV2*
	container *
shared_name *
dtype0*
shape:
* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
å
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AssignAssign3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsNencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0*
use_locking(
¤
8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/readIdentity3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0* 
_output_shapes
:

Ŗ
Iencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axisConst^encoder_1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
¶
Dencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concatConcatV29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2encoder_1/rnn/while/Identity_5Iencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*

Tidx0*
T0*
N* 
_output_shapes
:

 
Jencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
±
Dencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMulMatMulDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concatJencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
Ś
Dencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Initializer/ConstConst*
_output_shapes	
:*
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
valueB*    
ē
2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
Ó
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AssignAssign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesDencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Initializer/Const*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
validate_shape(*
_output_shapes	
:

7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/readIdentity2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:*
T0

Aencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/EnterEnter7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/read*
_output_shapes	
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAddBiasAddDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMulAencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter*
data_formatNHWC*
T0* 
_output_shapes
:

¤
Cencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dimConst^encoder_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
¤
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/splitSplitCencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split

9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add/yConst^encoder_1/rnn/while/Identity*
valueB
 *  ?*
_output_shapes
: *
dtype0
į
7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/addAdd;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:29encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add/y* 
_output_shapes
:
*
T0
Ŗ
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/SigmoidSigmoid7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add*
T0* 
_output_shapes
:

Ę
7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mulMul;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoidencoder_1/rnn/while/Identity_4*
T0* 
_output_shapes
:

®
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1Sigmoid9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split* 
_output_shapes
:
*
T0
Ø
8encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/TanhTanh;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:1*
T0* 
_output_shapes
:

ä
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1Mul=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_18encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0* 
_output_shapes
:

ß
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1Add7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1* 
_output_shapes
:
*
T0
°
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2Sigmoid;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:3* 
_output_shapes
:
*
T0
Ø
:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1Tanh9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1*
T0* 
_output_shapes
:

ę
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2Mul=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
T0* 
_output_shapes
:

Ų
&encoder_1/rnn/while/GreaterEqual/EnterEnterencoder_1/rnn/CheckSeqLen*
parallel_iterations *
T0*
_output_shapes	
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(

 encoder_1/rnn/while/GreaterEqualGreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:
Ł
 encoder_1/rnn/while/Select/EnterEnterencoder_1/rnn/zeros*(
_output_shapes
:’’’’’’’’’*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
Ī
encoder_1/rnn/while/SelectSelect encoder_1/rnn/while/GreaterEqual encoder_1/rnn/while/Select/Enter9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0* 
_output_shapes
:


"encoder_1/rnn/while/GreaterEqual_1GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
_output_shapes	
:*
T0
Š
encoder_1/rnn/while/Select_1Select"encoder_1/rnn/while/GreaterEqual_1encoder_1/rnn/while/Identity_29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1*
T0* 
_output_shapes
:


"encoder_1/rnn/while/GreaterEqual_2GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
_output_shapes	
:*
T0
Š
encoder_1/rnn/while/Select_2Select"encoder_1/rnn/while/GreaterEqual_2encoder_1/rnn/while/Identity_39encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
T0* 
_output_shapes
:


"encoder_1/rnn/while/GreaterEqual_3GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
_output_shapes	
:*
T0
Š
encoder_1/rnn/while/Select_3Select"encoder_1/rnn/while/GreaterEqual_3encoder_1/rnn/while/Identity_49encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1* 
_output_shapes
:
*
T0

"encoder_1/rnn/while/GreaterEqual_4GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:
Š
encoder_1/rnn/while/Select_4Select"encoder_1/rnn/while/GreaterEqual_4encoder_1/rnn/while/Identity_59encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0* 
_output_shapes
:


=encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterencoder_1/rnn/TensorArray*
is_constant(*
T0*,
_class"
 loc:@encoder_1/rnn/TensorArray*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
µ
7encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3=encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterencoder_1/rnn/while/Identityencoder_1/rnn/while/Selectencoder_1/rnn/while/Identity_1*,
_class"
 loc:@encoder_1/rnn/TensorArray*
_output_shapes
: *
T0
z
encoder_1/rnn/while/add/yConst^encoder_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
x
encoder_1/rnn/while/addAddencoder_1/rnn/while/Identityencoder_1/rnn/while/add/y*
T0*
_output_shapes
: 
l
!encoder_1/rnn/while/NextIterationNextIterationencoder_1/rnn/while/add*
_output_shapes
: *
T0

#encoder_1/rnn/while/NextIteration_1NextIteration7encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
}
#encoder_1/rnn/while/NextIteration_2NextIterationencoder_1/rnn/while/Select_1*
T0* 
_output_shapes
:

}
#encoder_1/rnn/while/NextIteration_3NextIterationencoder_1/rnn/while/Select_2* 
_output_shapes
:
*
T0
}
#encoder_1/rnn/while/NextIteration_4NextIterationencoder_1/rnn/while/Select_3*
T0* 
_output_shapes
:

}
#encoder_1/rnn/while/NextIteration_5NextIterationencoder_1/rnn/while/Select_4*
T0* 
_output_shapes
:

]
encoder_1/rnn/while/ExitExitencoder_1/rnn/while/Switch*
_output_shapes
: *
T0
c
encoder_1/rnn/while/Exit_1Exitencoder_1/rnn/while/Switch_1*
_output_shapes
:*
T0
s
encoder_1/rnn/while/Exit_2Exitencoder_1/rnn/while/Switch_2*
T0*(
_output_shapes
:’’’’’’’’’
s
encoder_1/rnn/while/Exit_3Exitencoder_1/rnn/while/Switch_3*(
_output_shapes
:’’’’’’’’’*
T0
s
encoder_1/rnn/while/Exit_4Exitencoder_1/rnn/while/Switch_4*(
_output_shapes
:’’’’’’’’’*
T0
s
encoder_1/rnn/while/Exit_5Exitencoder_1/rnn/while/Switch_5*
T0*(
_output_shapes
:’’’’’’’’’
Ā
0encoder_1/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3encoder_1/rnn/TensorArrayencoder_1/rnn/while/Exit_1*,
_class"
 loc:@encoder_1/rnn/TensorArray*
_output_shapes
: 

*encoder_1/rnn/TensorArrayStack/range/startConst*
value	B : *,
_class"
 loc:@encoder_1/rnn/TensorArray*
_output_shapes
: *
dtype0

*encoder_1/rnn/TensorArrayStack/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :*,
_class"
 loc:@encoder_1/rnn/TensorArray

$encoder_1/rnn/TensorArrayStack/rangeRange*encoder_1/rnn/TensorArrayStack/range/start0encoder_1/rnn/TensorArrayStack/TensorArraySizeV3*encoder_1/rnn/TensorArrayStack/range/delta*

Tidx0*#
_output_shapes
:’’’’’’’’’*,
_class"
 loc:@encoder_1/rnn/TensorArray
§
2encoder_1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3encoder_1/rnn/TensorArray$encoder_1/rnn/TensorArrayStack/rangeencoder_1/rnn/while/Exit_1*
element_shape:
*,
_class"
 loc:@encoder_1/rnn/TensorArray*
dtype0*%
_output_shapes
:
q
encoder_1/rnn/transpose/permConst*!
valueB"          *
_output_shapes
:*
dtype0
³
encoder_1/rnn/transpose	Transpose2encoder_1/rnn/TensorArrayStack/TensorArrayGatherV3encoder_1/rnn/transpose/perm*
Tperm0*
T0*%
_output_shapes
:
Æ
6output_projection/W/Initializer/truncated_normal/shapeConst*&
_class
loc:@output_projection/W*
valueB"      *
dtype0*
_output_shapes
:
¢
5output_projection/W/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *&
_class
loc:@output_projection/W*
valueB
 *    
¤
7output_projection/W/Initializer/truncated_normal/stddevConst*&
_class
loc:@output_projection/W*
valueB
 *ĶĢĢ=*
dtype0*
_output_shapes
: 

@output_projection/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6output_projection/W/Initializer/truncated_normal/shape*
_output_shapes
:	*
dtype0*
seed2æ*&
_class
loc:@output_projection/W*
T0*

seed{

4output_projection/W/Initializer/truncated_normal/mulMul@output_projection/W/Initializer/truncated_normal/TruncatedNormal7output_projection/W/Initializer/truncated_normal/stddev*
_output_shapes
:	*&
_class
loc:@output_projection/W*
T0
ö
0output_projection/W/Initializer/truncated_normalAdd4output_projection/W/Initializer/truncated_normal/mul5output_projection/W/Initializer/truncated_normal/mean*
T0*
_output_shapes
:	*&
_class
loc:@output_projection/W
±
output_projection/W
VariableV2*
shared_name *&
_class
loc:@output_projection/W*
	container *
shape:	*
dtype0*
_output_shapes
:	
ę
output_projection/W/AssignAssignoutput_projection/W0output_projection/W/Initializer/truncated_normal*
_output_shapes
:	*
validate_shape(*&
_class
loc:@output_projection/W*
T0*
use_locking(

output_projection/W/readIdentityoutput_projection/W*
T0*&
_class
loc:@output_projection/W*
_output_shapes
:	

%output_projection/b/Initializer/ConstConst*
dtype0*
_output_shapes
:*&
_class
loc:@output_projection/b*
valueB*ĶĢĢ=
§
output_projection/b
VariableV2*&
_class
loc:@output_projection/b*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Ö
output_projection/b/AssignAssignoutput_projection/b%output_projection/b/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*&
_class
loc:@output_projection/b

output_projection/b/readIdentityoutput_projection/b*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0
ŗ
"output_projection/xw_plus_b/MatMulMatMulencoder_1/rnn/while/Exit_4output_projection/W/read*
transpose_b( *
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( 
­
output_projection/xw_plus_bBiasAdd"output_projection/xw_plus_b/MatMuloutput_projection/b/read*
data_formatNHWC*
T0*'
_output_shapes
:’’’’’’’’’
s
output_projection/SoftmaxSoftmaxoutput_projection/xw_plus_b*
T0*'
_output_shapes
:’’’’’’’’’
d
"output_projection/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :

output_projection/ArgMaxArgMaxoutput_projection/xw_plus_b"output_projection/ArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:’’’’’’’’’
o

loss/ConstConst*
dtype0*
_output_shapes
:*1
value(B&"  ?¾z?`åp?bX?¤p}?  ?mē{>
j
loss/MulMuloutput_projection/xw_plus_b
loss/Const*'
_output_shapes
:’’’’’’’’’*
T0
X
	loss/CastCastcond/Merge_1*

SrcT0*
_output_shapes
:	*

DstT0
]
loss/logistic_loss/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
h
loss/logistic_loss/subSub
loss/Constloss/logistic_loss/sub/y*
_output_shapes
:*
T0
j
loss/logistic_loss/mulMulloss/logistic_loss/sub	loss/Cast*
_output_shapes
:	*
T0
]
loss/logistic_loss/add/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
y
loss/logistic_loss/addAddloss/logistic_loss/add/xloss/logistic_loss/mul*
_output_shapes
:	*
T0
_
loss/logistic_loss/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
p
loss/logistic_loss/sub_1Subloss/logistic_loss/sub_1/x	loss/Cast*
_output_shapes
:	*
T0
m
loss/logistic_loss/mul_1Mulloss/logistic_loss/sub_1loss/Mul*
_output_shapes
:	*
T0
Y
loss/logistic_loss/AbsAbsloss/Mul*'
_output_shapes
:’’’’’’’’’*
T0
g
loss/logistic_loss/NegNegloss/logistic_loss/Abs*'
_output_shapes
:’’’’’’’’’*
T0
g
loss/logistic_loss/ExpExploss/logistic_loss/Neg*
T0*'
_output_shapes
:’’’’’’’’’
k
loss/logistic_loss/Log1pLog1ploss/logistic_loss/Exp*'
_output_shapes
:’’’’’’’’’*
T0
[
loss/logistic_loss/Neg_1Negloss/Mul*'
_output_shapes
:’’’’’’’’’*
T0
k
loss/logistic_loss/ReluReluloss/logistic_loss/Neg_1*'
_output_shapes
:’’’’’’’’’*
T0

loss/logistic_loss/add_1Addloss/logistic_loss/Log1ploss/logistic_loss/Relu*'
_output_shapes
:’’’’’’’’’*
T0
{
loss/logistic_loss/mul_2Mulloss/logistic_loss/addloss/logistic_loss/add_1*
_output_shapes
:	*
T0
w
loss/logistic_lossAddloss/logistic_loss/mul_1loss/logistic_loss/mul_2*
T0*
_output_shapes
:	
]
loss/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
loss/SumSumloss/logistic_lossloss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
]
loss/Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
q
	loss/MeanMeanloss/logistic_lossloss/Const_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
[
accuracy/ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
t
accuracy/ArgMaxArgMaxcond/Merge_1accuracy/ArgMax/dimension*
_output_shapes	
:*
T0*

Tidx0
h
accuracy/EqualEqualoutput_projection/ArgMaxaccuracy/ArgMax*
T0	*
_output_shapes	
:
Z
accuracy/CastCastaccuracy/Equal*
_output_shapes	
:*

DstT0*

SrcT0

X
accuracy/ConstConst*
valueB: *
_output_shapes
:*
dtype0
v
accuracy/accuracyMeanaccuracy/Castaccuracy/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
`
learning_rate/learning_rateConst*
valueB
 *·Q8*
_output_shapes
: *
dtype0
Y
learning_rate/CastCastVariable/read*
_output_shapes
: *

DstT0*

SrcT0
Y
learning_rate/Cast_1/xConst*
_output_shapes
: *
dtype0*
value
B :'
d
learning_rate/Cast_1Castlearning_rate/Cast_1/x*

SrcT0*
_output_shapes
: *

DstT0
[
learning_rate/Cast_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *Āu?
k
learning_rate/truedivRealDivlearning_rate/Castlearning_rate/Cast_1*
_output_shapes
: *
T0
T
learning_rate/FloorFloorlearning_rate/truediv*
T0*
_output_shapes
: 
f
learning_rate/PowPowlearning_rate/Cast_2/xlearning_rate/Floor*
T0*
_output_shapes
: 
e
learning_rateMullearning_rate/learning_ratelearning_rate/Pow*
T0*
_output_shapes
: 
`
gradients/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
T
gradients/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
b
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
:	*
T0
S
gradients/f_countConst*
dtype0*
_output_shapes
: *
value	B : 
ø
gradients/f_count_1Entergradients/f_count*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
N*
T0*
_output_shapes
: : 
l
gradients/SwitchSwitchgradients/Mergeencoder_1/rnn/while/LoopCond*
_output_shapes
: : *
T0
p
gradients/Add/yConst^encoder_1/rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
_output_shapes
: *
T0
¹
gradients/NextIterationNextIterationgradients/AddA^gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPushA^gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPushA^gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPushA^gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPushZ^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPushb^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPushe^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPushZ^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPushb^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPushe^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPushc^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPush*
T0*
_output_shapes
: 
N
gradients/f_count_2Exitgradients/Switch*
T0*
_output_shapes
: 
S
gradients/b_countConst*
dtype0*
_output_shapes
: *
value	B :
Ä
gradients/b_count_1Entergradients/f_count_2*
parallel_iterations *
T0*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
_output_shapes
: : *
N*
T0
Ė
gradients/GreaterEqual/EnterEntergradients/b_count*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
x
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
O
gradients/b_count_2LoopCondgradients/GreaterEqual*
_output_shapes
: 
g
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
T0*
_output_shapes
: : 
i
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
_output_shapes
: *
T0

gradients/NextIteration_1NextIterationgradients/Sub>^gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/b_sync*
_output_shapes
: *
T0
P
gradients/b_count_3Exitgradients/Switch_1*
_output_shapes
: *
T0
x
'gradients/loss/logistic_loss_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
z
)gradients/loss/logistic_loss_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
į
7gradients/loss/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/loss/logistic_loss_grad/Shape)gradients/loss/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
µ
%gradients/loss/logistic_loss_grad/SumSumgradients/Fill7gradients/loss/logistic_loss_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¼
)gradients/loss/logistic_loss_grad/ReshapeReshape%gradients/loss/logistic_loss_grad/Sum'gradients/loss/logistic_loss_grad/Shape*
T0*
Tshape0*
_output_shapes
:	
¹
'gradients/loss/logistic_loss_grad/Sum_1Sumgradients/Fill9gradients/loss/logistic_loss_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ā
+gradients/loss/logistic_loss_grad/Reshape_1Reshape'gradients/loss/logistic_loss_grad/Sum_1)gradients/loss/logistic_loss_grad/Shape_1*
Tshape0*
_output_shapes
:	*
T0
~
-gradients/loss/logistic_loss/mul_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
w
/gradients/loss/logistic_loss/mul_1_grad/Shape_1Shapeloss/Mul*
T0*
out_type0*
_output_shapes
:
ó
=gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/mul_1_grad/Shape/gradients/loss/logistic_loss/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

+gradients/loss/logistic_loss/mul_1_grad/mulMul)gradients/loss/logistic_loss_grad/Reshapeloss/Mul*
_output_shapes
:	*
T0
Ž
+gradients/loss/logistic_loss/mul_1_grad/SumSum+gradients/loss/logistic_loss/mul_1_grad/mul=gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ī
/gradients/loss/logistic_loss/mul_1_grad/ReshapeReshape+gradients/loss/logistic_loss/mul_1_grad/Sum-gradients/loss/logistic_loss/mul_1_grad/Shape*
Tshape0*
_output_shapes
:	*
T0
£
-gradients/loss/logistic_loss/mul_1_grad/mul_1Mulloss/logistic_loss/sub_1)gradients/loss/logistic_loss_grad/Reshape*
_output_shapes
:	*
T0
ä
-gradients/loss/logistic_loss/mul_1_grad/Sum_1Sum-gradients/loss/logistic_loss/mul_1_grad/mul_1?gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ü
1gradients/loss/logistic_loss/mul_1_grad/Reshape_1Reshape-gradients/loss/logistic_loss/mul_1_grad/Sum_1/gradients/loss/logistic_loss/mul_1_grad/Shape_1*
T0*'
_output_shapes
:’’’’’’’’’*
Tshape0
~
-gradients/loss/logistic_loss/mul_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      

/gradients/loss/logistic_loss/mul_2_grad/Shape_1Shapeloss/logistic_loss/add_1*
T0*
out_type0*
_output_shapes
:
ó
=gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/mul_2_grad/Shape/gradients/loss/logistic_loss/mul_2_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
£
+gradients/loss/logistic_loss/mul_2_grad/mulMul+gradients/loss/logistic_loss_grad/Reshape_1loss/logistic_loss/add_1*
T0*
_output_shapes
:	
Ž
+gradients/loss/logistic_loss/mul_2_grad/SumSum+gradients/loss/logistic_loss/mul_2_grad/mul=gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ī
/gradients/loss/logistic_loss/mul_2_grad/ReshapeReshape+gradients/loss/logistic_loss/mul_2_grad/Sum-gradients/loss/logistic_loss/mul_2_grad/Shape*
Tshape0*
_output_shapes
:	*
T0
£
-gradients/loss/logistic_loss/mul_2_grad/mul_1Mulloss/logistic_loss/add+gradients/loss/logistic_loss_grad/Reshape_1*
_output_shapes
:	*
T0
ä
-gradients/loss/logistic_loss/mul_2_grad/Sum_1Sum-gradients/loss/logistic_loss/mul_2_grad/mul_1?gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ü
1gradients/loss/logistic_loss/mul_2_grad/Reshape_1Reshape-gradients/loss/logistic_loss/mul_2_grad/Sum_1/gradients/loss/logistic_loss/mul_2_grad/Shape_1*'
_output_shapes
:’’’’’’’’’*
Tshape0*
T0

-gradients/loss/logistic_loss/add_1_grad/ShapeShapeloss/logistic_loss/Log1p*
T0*
_output_shapes
:*
out_type0

/gradients/loss/logistic_loss/add_1_grad/Shape_1Shapeloss/logistic_loss/Relu*
_output_shapes
:*
out_type0*
T0
ó
=gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/add_1_grad/Shape/gradients/loss/logistic_loss/add_1_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
ä
+gradients/loss/logistic_loss/add_1_grad/SumSum1gradients/loss/logistic_loss/mul_2_grad/Reshape_1=gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ö
/gradients/loss/logistic_loss/add_1_grad/ReshapeReshape+gradients/loss/logistic_loss/add_1_grad/Sum-gradients/loss/logistic_loss/add_1_grad/Shape*'
_output_shapes
:’’’’’’’’’*
Tshape0*
T0
č
-gradients/loss/logistic_loss/add_1_grad/Sum_1Sum1gradients/loss/logistic_loss/mul_2_grad/Reshape_1?gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ü
1gradients/loss/logistic_loss/add_1_grad/Reshape_1Reshape-gradients/loss/logistic_loss/add_1_grad/Sum_1/gradients/loss/logistic_loss/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
¤
-gradients/loss/logistic_loss/Log1p_grad/add/xConst0^gradients/loss/logistic_loss/add_1_grad/Reshape*
valueB
 *  ?*
_output_shapes
: *
dtype0
«
+gradients/loss/logistic_loss/Log1p_grad/addAdd-gradients/loss/logistic_loss/Log1p_grad/add/xloss/logistic_loss/Exp*'
_output_shapes
:’’’’’’’’’*
T0

2gradients/loss/logistic_loss/Log1p_grad/Reciprocal
Reciprocal+gradients/loss/logistic_loss/Log1p_grad/add*
T0*'
_output_shapes
:’’’’’’’’’
É
+gradients/loss/logistic_loss/Log1p_grad/mulMul/gradients/loss/logistic_loss/add_1_grad/Reshape2gradients/loss/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:’’’’’’’’’*
T0
¹
/gradients/loss/logistic_loss/Relu_grad/ReluGradReluGrad1gradients/loss/logistic_loss/add_1_grad/Reshape_1loss/logistic_loss/Relu*
T0*'
_output_shapes
:’’’’’’’’’
§
)gradients/loss/logistic_loss/Exp_grad/mulMul+gradients/loss/logistic_loss/Log1p_grad/mulloss/logistic_loss/Exp*'
_output_shapes
:’’’’’’’’’*
T0

+gradients/loss/logistic_loss/Neg_1_grad/NegNeg/gradients/loss/logistic_loss/Relu_grad/ReluGrad*
T0*'
_output_shapes
:’’’’’’’’’

)gradients/loss/logistic_loss/Neg_grad/NegNeg)gradients/loss/logistic_loss/Exp_grad/mul*'
_output_shapes
:’’’’’’’’’*
T0
n
*gradients/loss/logistic_loss/Abs_grad/SignSignloss/Mul*'
_output_shapes
:’’’’’’’’’*
T0
¹
)gradients/loss/logistic_loss/Abs_grad/mulMul)gradients/loss/logistic_loss/Neg_grad/Neg*gradients/loss/logistic_loss/Abs_grad/Sign*'
_output_shapes
:’’’’’’’’’*
T0
¢
gradients/AddNAddN1gradients/loss/logistic_loss/mul_1_grad/Reshape_1+gradients/loss/logistic_loss/Neg_1_grad/Neg)gradients/loss/logistic_loss/Abs_grad/mul*
N*
T0*'
_output_shapes
:’’’’’’’’’*D
_class:
86loc:@gradients/loss/logistic_loss/mul_1_grad/Reshape_1
x
gradients/loss/Mul_grad/ShapeShapeoutput_projection/xw_plus_b*
_output_shapes
:*
out_type0*
T0
i
gradients/loss/Mul_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
Ć
-gradients/loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/Mul_grad/Shapegradients/loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
p
gradients/loss/Mul_grad/mulMulgradients/AddN
loss/Const*
T0*'
_output_shapes
:’’’’’’’’’
®
gradients/loss/Mul_grad/SumSumgradients/loss/Mul_grad/mul-gradients/loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¦
gradients/loss/Mul_grad/ReshapeReshapegradients/loss/Mul_grad/Sumgradients/loss/Mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

gradients/loss/Mul_grad/mul_1Muloutput_projection/xw_plus_bgradients/AddN*
T0*'
_output_shapes
:’’’’’’’’’
“
gradients/loss/Mul_grad/Sum_1Sumgradients/loss/Mul_grad/mul_1/gradients/loss/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

!gradients/loss/Mul_grad/Reshape_1Reshapegradients/loss/Mul_grad/Sum_1gradients/loss/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
¢
6gradients/output_projection/xw_plus_b_grad/BiasAddGradBiasAddGradgradients/loss/Mul_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
Ö
8gradients/output_projection/xw_plus_b/MatMul_grad/MatMulMatMulgradients/loss/Mul_grad/Reshapeoutput_projection/W/read*
transpose_b(*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
T0
Ń
:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1MatMulencoder_1/rnn/while/Exit_4gradients/loss/Mul_grad/Reshape*
transpose_b( *
T0*
_output_shapes
:	*
transpose_a(
`
gradients/zeros_like	ZerosLikeencoder_1/rnn/while/Exit_1*
T0*
_output_shapes
:
r
gradients/zeros_like_1	ZerosLikeencoder_1/rnn/while/Exit_2*(
_output_shapes
:’’’’’’’’’*
T0
r
gradients/zeros_like_2	ZerosLikeencoder_1/rnn/while/Exit_3*
T0*(
_output_shapes
:’’’’’’’’’
r
gradients/zeros_like_3	ZerosLikeencoder_1/rnn/while/Exit_5*(
_output_shapes
:’’’’’’’’’*
T0

0gradients/encoder_1/rnn/while/Exit_4_grad/b_exitEnter8gradients/output_projection/xw_plus_b/MatMul_grad/MatMul*
is_constant( *(
_output_shapes
:’’’’’’’’’*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
ä
0gradients/encoder_1/rnn/while/Exit_1_grad/b_exitEntergradients/zeros_like*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
ö
0gradients/encoder_1/rnn/while/Exit_2_grad/b_exitEntergradients/zeros_like_1*
parallel_iterations *
T0*(
_output_shapes
:’’’’’’’’’*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
ö
0gradients/encoder_1/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_like_2*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:’’’’’’’’’*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
ö
0gradients/encoder_1/rnn/while/Exit_5_grad/b_exitEntergradients/zeros_like_3*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:’’’’’’’’’*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
ź
4gradients/encoder_1/rnn/while/Switch_4_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_4_grad/b_exit;gradients/encoder_1/rnn/while/Switch_4_grad_1/NextIteration**
_output_shapes
:’’’’’’’’’: *
N*
T0
ź
4gradients/encoder_1/rnn/while/Switch_2_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_2_grad/b_exit;gradients/encoder_1/rnn/while/Switch_2_grad_1/NextIteration**
_output_shapes
:’’’’’’’’’: *
N*
T0
ź
4gradients/encoder_1/rnn/while/Switch_3_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_3_grad/b_exit;gradients/encoder_1/rnn/while/Switch_3_grad_1/NextIteration**
_output_shapes
:’’’’’’’’’: *
N*
T0
ź
4gradients/encoder_1/rnn/while/Switch_5_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_5_grad/b_exit;gradients/encoder_1/rnn/while/Switch_5_grad_1/NextIteration**
_output_shapes
:’’’’’’’’’: *
N*
T0

1gradients/encoder_1/rnn/while/Merge_4_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*4
_output_shapes"
 :’’’’’’’’’:
*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_4_grad/b_switch*
T0

1gradients/encoder_1/rnn/while/Merge_2_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*
T0*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_2_grad/b_switch*4
_output_shapes"
 :’’’’’’’’’:


1gradients/encoder_1/rnn/while/Merge_3_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_3_grad/b_switch*4
_output_shapes"
 :’’’’’’’’’:
*
T0

1gradients/encoder_1/rnn/while/Merge_5_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_5_grad/b_switchgradients/b_count_2*4
_output_shapes"
 :’’’’’’’’’:
*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_5_grad/b_switch*
T0

/gradients/encoder_1/rnn/while/Enter_4_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_4_grad/Switch*
T0*(
_output_shapes
:’’’’’’’’’

/gradients/encoder_1/rnn/while/Enter_2_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_2_grad/Switch*
T0*(
_output_shapes
:’’’’’’’’’

/gradients/encoder_1/rnn/while/Enter_3_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_3_grad/Switch*(
_output_shapes
:’’’’’’’’’*
T0

/gradients/encoder_1/rnn/while/Enter_5_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_5_grad/Switch*
T0*(
_output_shapes
:’’’’’’’’’

4gradients/encoder_1/initial_state_2_tiled_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      
Ū
4gradients/encoder_1/initial_state_2_tiled_grad/stackPack)encoder_1/initial_state_2_tiled/multiples4gradients/encoder_1/initial_state_2_tiled_grad/Shape*
T0*

axis *
N*
_output_shapes

:

=gradients/encoder_1/initial_state_2_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_2_tiled_grad/stack*
T0*
_output_shapes
: 

>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
ć
<gradients/encoder_1/initial_state_2_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_2_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub/y*
T0*
_output_shapes
: 

Dgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/startConst*
_output_shapes
: *
dtype0*
value	B : 

Dgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
ŗ
>gradients/encoder_1/initial_state_2_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_2_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/delta*

Tidx0*
_output_shapes
:
č
>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_2_tiled_grad/transpose/Range*
_output_shapes
:*
T0
ń
8gradients/encoder_1/initial_state_2_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_2_tiled_grad/stack>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub_1*
Tperm0*
_output_shapes

:*
T0

<gradients/encoder_1/initial_state_2_tiled_grad/Reshape/shapeConst*
valueB:
’’’’’’’’’*
_output_shapes
:*
dtype0
ģ
6gradients/encoder_1/initial_state_2_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_2_tiled_grad/transpose<gradients/encoder_1/initial_state_2_tiled_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
u
3gradients/encoder_1/initial_state_2_tiled_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
|
:gradients/encoder_1/initial_state_2_tiled_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
|
:gradients/encoder_1/initial_state_2_tiled_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0

4gradients/encoder_1/initial_state_2_tiled_grad/rangeRange:gradients/encoder_1/initial_state_2_tiled_grad/range/start3gradients/encoder_1/initial_state_2_tiled_grad/Size:gradients/encoder_1/initial_state_2_tiled_grad/range/delta*
_output_shapes
:*

Tidx0

8gradients/encoder_1/initial_state_2_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_4_grad/Exit6gradients/encoder_1/initial_state_2_tiled_grad/Reshape*
Tshape0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
T0
š
2gradients/encoder_1/initial_state_2_tiled_grad/SumSum8gradients/encoder_1/initial_state_2_tiled_grad/Reshape_14gradients/encoder_1/initial_state_2_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	
·
<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4
É
?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4
§
@gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_4^gradients/Add*
_output_shapes
:*
swap_memory( *1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
T0
Ü
Hgradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*(
_output_shapes
:’’’’’’’’’

=gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/b_syncControlTrigger@^gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopa^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopd^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopa^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopd^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopb^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop
·
6gradients/encoder_1/rnn/while/Select_3_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop*(
_output_shapes
:’’’’’’’’’*
T0
·
8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_accStack*
	elem_type0
*
_output_shapes
:*

stack_name *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3
Å
;gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3
§
<gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_3^gradients/Add*
T0
*
_output_shapes
:*
swap_memory( *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3
Ų
Dgradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
T0

;gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*
_output_shapes	
:*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3

2gradients/encoder_1/rnn/while/Select_3_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_4_grad/Switch:16gradients/encoder_1/rnn/while/Select_3_grad/zeros_like*
T0* 
_output_shapes
:


4gradients/encoder_1/rnn/while/Select_3_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_3_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_4_grad/Switch:1* 
_output_shapes
:
*
T0

4gradients/encoder_1/initial_state_0_tiled_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
Ū
4gradients/encoder_1/initial_state_0_tiled_grad/stackPack)encoder_1/initial_state_0_tiled/multiples4gradients/encoder_1/initial_state_0_tiled_grad/Shape*
T0*

axis *
N*
_output_shapes

:

=gradients/encoder_1/initial_state_0_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_0_tiled_grad/stack*
T0*
_output_shapes
: 

>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub/yConst*
value	B :*
_output_shapes
: *
dtype0
ć
<gradients/encoder_1/initial_state_0_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_0_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub/y*
_output_shapes
: *
T0

Dgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/startConst*
value	B : *
_output_shapes
: *
dtype0

Dgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ŗ
>gradients/encoder_1/initial_state_0_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_0_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/delta*
_output_shapes
:*

Tidx0
č
>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_0_tiled_grad/transpose/Range*
_output_shapes
:*
T0
ń
8gradients/encoder_1/initial_state_0_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_0_tiled_grad/stack>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub_1*
Tperm0*
_output_shapes

:*
T0

<gradients/encoder_1/initial_state_0_tiled_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
’’’’’’’’’
ģ
6gradients/encoder_1/initial_state_0_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_0_tiled_grad/transpose<gradients/encoder_1/initial_state_0_tiled_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
u
3gradients/encoder_1/initial_state_0_tiled_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
|
:gradients/encoder_1/initial_state_0_tiled_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
|
:gradients/encoder_1/initial_state_0_tiled_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

4gradients/encoder_1/initial_state_0_tiled_grad/rangeRange:gradients/encoder_1/initial_state_0_tiled_grad/range/start3gradients/encoder_1/initial_state_0_tiled_grad/Size:gradients/encoder_1/initial_state_0_tiled_grad/range/delta*
_output_shapes
:*

Tidx0

8gradients/encoder_1/initial_state_0_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_2_grad/Exit6gradients/encoder_1/initial_state_0_tiled_grad/Reshape*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
Tshape0
š
2gradients/encoder_1/initial_state_0_tiled_grad/SumSum8gradients/encoder_1/initial_state_0_tiled_grad/Reshape_14gradients/encoder_1/initial_state_0_tiled_grad/range*
_output_shapes
:	*
T0*
	keep_dims( *

Tidx0
·
<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2
É
?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2
§
@gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_2^gradients/Add*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*
_output_shapes
:*
swap_memory( *
T0
Ü
Hgradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*(
_output_shapes
:’’’’’’’’’*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2
·
6gradients/encoder_1/rnn/while/Select_1_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop*(
_output_shapes
:’’’’’’’’’*
T0
·
8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_accStack*
	elem_type0
*
_output_shapes
:*

stack_name *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1
Å
;gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
T0
§
<gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_1^gradients/Add*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
_output_shapes
:*
swap_memory( *
T0

Ų
Dgradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
_output_shapes	
:

2gradients/encoder_1/rnn/while/Select_1_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_2_grad/Switch:16gradients/encoder_1/rnn/while/Select_1_grad/zeros_like* 
_output_shapes
:
*
T0

4gradients/encoder_1/rnn/while/Select_1_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_1_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_2_grad/Switch:1*
T0* 
_output_shapes
:


4gradients/encoder_1/initial_state_1_tiled_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      
Ū
4gradients/encoder_1/initial_state_1_tiled_grad/stackPack)encoder_1/initial_state_1_tiled/multiples4gradients/encoder_1/initial_state_1_tiled_grad/Shape*
_output_shapes

:*
N*

axis *
T0

=gradients/encoder_1/initial_state_1_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_1_tiled_grad/stack*
_output_shapes
: *
T0

>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub/yConst*
value	B :*
_output_shapes
: *
dtype0
ć
<gradients/encoder_1/initial_state_1_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_1_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub/y*
T0*
_output_shapes
: 

Dgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 

Dgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
ŗ
>gradients/encoder_1/initial_state_1_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_1_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/delta*

Tidx0*
_output_shapes
:
č
>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_1_tiled_grad/transpose/Range*
T0*
_output_shapes
:
ń
8gradients/encoder_1/initial_state_1_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_1_tiled_grad/stack>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub_1*
Tperm0*
T0*
_output_shapes

:

<gradients/encoder_1/initial_state_1_tiled_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’
ģ
6gradients/encoder_1/initial_state_1_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_1_tiled_grad/transpose<gradients/encoder_1/initial_state_1_tiled_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
u
3gradients/encoder_1/initial_state_1_tiled_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
|
:gradients/encoder_1/initial_state_1_tiled_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
|
:gradients/encoder_1/initial_state_1_tiled_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0

4gradients/encoder_1/initial_state_1_tiled_grad/rangeRange:gradients/encoder_1/initial_state_1_tiled_grad/range/start3gradients/encoder_1/initial_state_1_tiled_grad/Size:gradients/encoder_1/initial_state_1_tiled_grad/range/delta*

Tidx0*
_output_shapes
:

8gradients/encoder_1/initial_state_1_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_3_grad/Exit6gradients/encoder_1/initial_state_1_tiled_grad/Reshape*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
Tshape0
š
2gradients/encoder_1/initial_state_1_tiled_grad/SumSum8gradients/encoder_1/initial_state_1_tiled_grad/Reshape_14gradients/encoder_1/initial_state_1_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	
·
<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_accStack*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*

stack_name *
_output_shapes
:
É
?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3
§
@gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_3^gradients/Add*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*
_output_shapes
:*
swap_memory( *
T0
Ü
Hgradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*(
_output_shapes
:’’’’’’’’’*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3
·
6gradients/encoder_1/rnn/while/Select_2_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop*
T0*(
_output_shapes
:’’’’’’’’’
·
8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
_output_shapes
:*

stack_name 
Å
;gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
§
<gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_2^gradients/Add*
_output_shapes
:*
swap_memory( *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
T0

Ų
Dgradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2

;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
_output_shapes	
:

2gradients/encoder_1/rnn/while/Select_2_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_3_grad/Switch:16gradients/encoder_1/rnn/while/Select_2_grad/zeros_like* 
_output_shapes
:
*
T0

4gradients/encoder_1/rnn/while/Select_2_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_2_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_3_grad/Switch:1* 
_output_shapes
:
*
T0

4gradients/encoder_1/initial_state_3_tiled_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      
Ū
4gradients/encoder_1/initial_state_3_tiled_grad/stackPack)encoder_1/initial_state_3_tiled/multiples4gradients/encoder_1/initial_state_3_tiled_grad/Shape*

axis *
_output_shapes

:*
T0*
N

=gradients/encoder_1/initial_state_3_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_3_tiled_grad/stack*
_output_shapes
: *
T0

>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub/yConst*
value	B :*
_output_shapes
: *
dtype0
ć
<gradients/encoder_1/initial_state_3_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_3_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub/y*
T0*
_output_shapes
: 

Dgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/startConst*
value	B : *
_output_shapes
: *
dtype0

Dgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ŗ
>gradients/encoder_1/initial_state_3_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_3_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/delta*
_output_shapes
:*

Tidx0
č
>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_3_tiled_grad/transpose/Range*
T0*
_output_shapes
:
ń
8gradients/encoder_1/initial_state_3_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_3_tiled_grad/stack>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub_1*
Tperm0*
T0*
_output_shapes

:

<gradients/encoder_1/initial_state_3_tiled_grad/Reshape/shapeConst*
valueB:
’’’’’’’’’*
_output_shapes
:*
dtype0
ģ
6gradients/encoder_1/initial_state_3_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_3_tiled_grad/transpose<gradients/encoder_1/initial_state_3_tiled_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
u
3gradients/encoder_1/initial_state_3_tiled_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
|
:gradients/encoder_1/initial_state_3_tiled_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
|
:gradients/encoder_1/initial_state_3_tiled_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0

4gradients/encoder_1/initial_state_3_tiled_grad/rangeRange:gradients/encoder_1/initial_state_3_tiled_grad/range/start3gradients/encoder_1/initial_state_3_tiled_grad/Size:gradients/encoder_1/initial_state_3_tiled_grad/range/delta*

Tidx0*
_output_shapes
:

8gradients/encoder_1/initial_state_3_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_5_grad/Exit6gradients/encoder_1/initial_state_3_tiled_grad/Reshape*
T0*
Tshape0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
š
2gradients/encoder_1/initial_state_3_tiled_grad/SumSum8gradients/encoder_1/initial_state_3_tiled_grad/Reshape_14gradients/encoder_1/initial_state_3_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	
·
<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *1
_class'
%#loc:@encoder_1/rnn/while/Identity_5
É
?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc*
is_constant(*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
§
@gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_5^gradients/Add*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*
_output_shapes
:*
swap_memory( 
Ü
Hgradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5

?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*(
_output_shapes
:’’’’’’’’’*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5
·
6gradients/encoder_1/rnn/while/Select_4_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop*
T0*(
_output_shapes
:’’’’’’’’’
·
8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*

stack_name *
_output_shapes
:
Å
;gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
§
<gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_4^gradients/Add*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
_output_shapes
:*
swap_memory( *
T0

Ų
Dgradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc*
is_constant(*
T0*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 

;gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*
_output_shapes	
:*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4

2gradients/encoder_1/rnn/while/Select_4_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_5_grad/Switch:16gradients/encoder_1/rnn/while/Select_4_grad/zeros_like*
T0* 
_output_shapes
:


4gradients/encoder_1/rnn/while/Select_4_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_4_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_5_grad/Switch:1* 
_output_shapes
:
*
T0
Æ
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
±
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
é
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnter:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1^gradients/Add*
_output_shapes
:*
swap_memory( *M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
T0
¤
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
Ó
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1* 
_output_shapes
:


Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mulMul4gradients/encoder_1/rnn/while/Select_4_grad/Select_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop* 
_output_shapes
:
*
T0
Į
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape*
T0* 
_output_shapes
:
*
Tshape0
ī
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_accStack*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2^gradients/Add*
_output_shapes
:*
swap_memory( *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
T0
«
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2
Ś
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2* 
_output_shapes
:


Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop4gradients/encoder_1/rnn/while/Select_4_grad/Select_1* 
_output_shapes
:
*
T0
Ē
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ø
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*
T0* 
_output_shapes
:
*
Tshape0
½
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape* 
_output_shapes
:
*
T0
“
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1*
T0* 
_output_shapes
:


gradients/AddN_1AddN4gradients/encoder_1/rnn/while/Select_3_grad/Select_1Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGrad* 
_output_shapes
:
*
N*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_3_grad/Select_1*
T0
Æ
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
±
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/SumSumgradients/AddN_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape* 
_output_shapes
:
*
Tshape0*
T0

Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ø
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1* 
_output_shapes
:
*
Tshape0*
T0
­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ShapeConst^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
¬
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1Shapeencoder_1/rnn/while/Identity_4*
T0*
_output_shapes
:*
out_type0

bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1
Å
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
is_constant(*
T0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
Ó
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1^gradients/Add*
_output_shapes
:*
swap_memory( *a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
T0
Ų
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/

egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*
	elem_type0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
_output_shapes
:
ē
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shapeegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mulMulPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop* 
_output_shapes
:
*
T0
»
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumSumJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¬
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0
ź
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnter;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid^gradients/Add*
_output_shapes
:*
swap_memory( *N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
T0
„
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
T0*
is_constant(
Ō
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid
§
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1MulUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape* 
_output_shapes
:
*
T0
Į
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ń
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*(
_output_shapes
:’’’’’’’’’*
Tshape0*
T0
Æ
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
±
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
ē
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnter8encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh^gradients/Add*
T0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
_output_shapes
:*
swap_memory( 
¢
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0
Ń
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh
©
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mulMulRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop* 
_output_shapes
:
*
T0
Į
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0
ī
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
T0*
is_constant(

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1^gradients/Add*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
_output_shapes
:*
swap_memory( 
«
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
Ś
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1
­
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1*
T0* 
_output_shapes
:

Ē
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ø
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1* 
_output_shapes
:
*
Tshape0*
T0
·
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape*
T0* 
_output_shapes
:


gradients/AddN_2AddN2gradients/encoder_1/rnn/while/Select_3_grad/SelectPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1*
T0*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_3_grad/Select*
N* 
_output_shapes
:

½
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape*
T0* 
_output_shapes
:

²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1*
T0* 
_output_shapes
:

­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
”
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
Š
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
Ē
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumSumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¬
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape*
T0*
Tshape0* 
_output_shapes
:

Ė
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1SumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ø
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0

;gradients/encoder_1/rnn/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_2*
T0* 
_output_shapes
:

õ
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_accStack*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
_output_shapes
:*

stack_name 
 
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim
£
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPush	StackPushXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnterCencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim
³
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc*
T0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
Ų
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPopStackPopagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop/RefEnter^gradients/Sub*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
_output_shapes
: 
Ė
Ogradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concatConcatV2Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop* 
_output_shapes
:
*
N*
T0*

Tidx0
ó
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:
Ą
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read*
parallel_iterations *
T0* 
_output_shapes
:
*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(
č
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMulMatMulOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
T0* 
_output_shapes
:
*
transpose_a( 

bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat
»
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat
æ
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat^gradients/Add*
T0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
_output_shapes
:*
swap_memory( 
Ī
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
ż
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat* 
_output_shapes
:

ļ
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1MatMulegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
„
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
_output_shapes	
:*
valueB*    
Ń
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes	
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
Ģ
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIteration*
_output_shapes
	:: *
T0*
N
ż
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*"
_output_shapes
::*
T0
“
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/AddAddYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch:1Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
ė
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
ß
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:*
T0
Ŗ
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
_output_shapes
: *
dtype0

]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_accStack*
	elem_type0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*

stack_name 
¶
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis
æ
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPush	StackPush`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnterIencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis
É
igradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
ī
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopStackPopigradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnter^gradients/Sub*
	elem_type0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
_output_shapes
: 
Ą
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/modFloorMod`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
ŗ
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
¹
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Shape_1Shapeencoder_1/rnn/while/Identity_5*
T0*
_output_shapes
:*
out_type0
ö
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2
¬
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
T0
„
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPush	StackPushcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2^gradients/Add*
T0*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:*
swap_memory( 
æ
lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
ī
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopStackPoplgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2
Ī
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeNShapeNcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop* 
_output_shapes
::*
N*
out_type0*
T0
®
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffsetConcatOffsetWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/modZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
“
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/SliceSliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffsetZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN* 
_output_shapes
:
*
Index0*
T0
Ā
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice_1SliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMulbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffset:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:’’’’’’’’’
ø
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
*    *
dtype0* 
_output_shapes
:

č
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1Enter_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc*
is_constant( * 
_output_shapes
:
*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
ģ
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2Mergeagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
: 

`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/SwitchSwitchagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
:

Ń
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/AddAddbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:


ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIterationNextIteration]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:

ö
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3Exit`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
*
T0
¦
gradients/AddN_3AddN4gradients/encoder_1/rnn/while/Select_2_grad/Select_1Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_2_grad/Select_1* 
_output_shapes
:
*
T0*
N
Æ
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
±
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
é
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnter:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1^gradients/Add*
_output_shapes
:*
swap_memory( *M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0
¤
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0*
is_constant(
Ó
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1* 
_output_shapes
:

ē
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mulMulgradients/AddN_3Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop* 
_output_shapes
:
*
T0
Į
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape*
T0*
Tshape0* 
_output_shapes
:

ī
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_accStack*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
_output_shapes
:*

stack_name 

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2
«
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
Ś
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2
ė
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopgradients/AddN_3* 
_output_shapes
:
*
T0
Ē
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ø
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*
Tshape0* 
_output_shapes
:
*
T0
¤
gradients/AddN_4AddN2gradients/encoder_1/rnn/while/Select_4_grad/Select[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice_1*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_4_grad/Select* 
_output_shapes
:
*
T0*
N
½
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape* 
_output_shapes
:
*
T0
“
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1* 
_output_shapes
:
*
T0

;gradients/encoder_1/rnn/while/Switch_5_grad_1/NextIterationNextIterationgradients/AddN_4* 
_output_shapes
:
*
T0

gradients/AddN_5AddN4gradients/encoder_1/rnn/while/Select_1_grad/Select_1Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGrad* 
_output_shapes
:
*
N*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_1_grad/Select_1*
T0
Æ
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
±
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/SumSumgradients/AddN_5^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0

Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_5`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ø
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*
Tshape0* 
_output_shapes
:
*
T0
­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
¬
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1Shapeencoder_1/rnn/while/Identity_2*
T0*
_output_shapes
:*
out_type0

bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1
Å
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
T0*
is_constant(
Ó
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1^gradients/Add*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*
swap_memory( *
T0
Ų
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1

egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*
	elem_type0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:
ē
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shapeegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mulMulPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop* 
_output_shapes
:
*
T0
»
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumSumJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¬
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape*
T0* 
_output_shapes
:
*
Tshape0
ź
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
T0*
is_constant(

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnter;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid^gradients/Add*
_output_shapes
:*
swap_memory( *N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
T0
„
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
Ō
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid* 
_output_shapes
:

§
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1MulUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape* 
_output_shapes
:
*
T0
Į
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ń
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*(
_output_shapes
:’’’’’’’’’*
Tshape0
Æ
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
±
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
ē
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_accStack*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
_output_shapes
:*

stack_name 

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc*
T0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnter8encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh^gradients/Add*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
_output_shapes
:*
swap_memory( *
T0
¢
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
Ń
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh
©
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mulMulRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop*
T0* 
_output_shapes
:

Į
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape* 
_output_shapes
:
*
Tshape0*
T0
ī
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_accStack*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
_output_shapes
:*

stack_name 

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
T0*
is_constant(

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1^gradients/Add*
_output_shapes
:*
swap_memory( *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
T0
«
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc*
is_constant(*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
Ś
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1* 
_output_shapes
:

­
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1*
T0* 
_output_shapes
:

Ē
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ø
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*
Tshape0* 
_output_shapes
:
*
T0
·
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape*
T0* 
_output_shapes
:


gradients/AddN_6AddN2gradients/encoder_1/rnn/while/Select_1_grad/SelectPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1*
T0*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_1_grad/Select*
N* 
_output_shapes
:

½
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape*
T0* 
_output_shapes
:

²
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1*
T0* 
_output_shapes
:

­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
”
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
_output_shapes
: *
dtype0*
valueB 
Š
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
Ē
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumSumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¬
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape* 
_output_shapes
:
*
Tshape0*
T0
Ė
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1SumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ø
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0

;gradients/encoder_1/rnn/while/Switch_2_grad_1/NextIterationNextIterationgradients/AddN_6* 
_output_shapes
:
*
T0
õ
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_accStack*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*

stack_name *
_output_shapes
:
 
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
T0*
is_constant(
£
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPush	StackPushXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnterCencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim^gradients/Add*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
_output_shapes
:*
swap_memory( *
T0
³
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim
Ų
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPopStackPopagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
: *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim
Ė
Ogradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concatConcatV2Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop* 
_output_shapes
:
*
N*
T0*

Tidx0
ó
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
data_formatNHWC*
T0*
_output_shapes	
:
Ą
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
č
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMulMatMulOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(* 
_output_shapes
:
*
transpose_a( *
T0

bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
»
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat*
T0
æ
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat^gradients/Add*
T0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat*
_output_shapes
:*
swap_memory( 
Ī
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
ż
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
ļ
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1MatMulegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
transpose_b( * 
_output_shapes
:
*
transpose_a(*
T0
„
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
_output_shapes	
:*
dtype0
Ń
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc*
parallel_iterations *
T0*
_output_shapes	
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
Ģ
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:: 
ż
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
::
“
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/AddAddYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch:1Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
ė
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:*
T0
ß
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:*
T0
Ŗ
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis
¶
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
is_constant(*
T0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
æ
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPush	StackPush`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnterIencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis
É
igradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis
ī
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopStackPopigradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnter^gradients/Sub*
	elem_type0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
_output_shapes
: 
Ą
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/modFloorMod`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
ŗ
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
¹
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Shape_1Shapeencoder_1/rnn/while/Identity_3*
T0*
_output_shapes
:*
out_type0
Ų
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *.
_class$
" loc:@encoder_1/rnn/TensorArray_1

cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
is_constant(*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
ó
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPush	StackPushcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter%encoder_1/rnn/while/TensorArrayReadV3^gradients/Add*
_output_shapes
:*
swap_memory( *.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
T0
”
lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
Š
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopStackPoplgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
Ī
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeNShapeNcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop*
out_type0* 
_output_shapes
::*
T0*
N
®
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffsetConcatOffsetWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/modZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
“
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/SliceSliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffsetZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN*
Index0*
T0* 
_output_shapes
:

Ā
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Slice_1SliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMulbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffset:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:’’’’’’’’’
ø
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_accConst* 
_output_shapes
:
*
dtype0*
valueB
*    
č
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1Enter_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc*
is_constant( * 
_output_shapes
:
*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
ģ
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2Mergeagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
: 

`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/SwitchSwitchagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
:

Ń
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/AddAddbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0

ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIterationNextIteration]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
*
T0
ö
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3Exit`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:

É
\gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterencoder_1/rnn/TensorArray_1*
is_constant(*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
ō
^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterHencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
T0
 
Vgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3\gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*
_output_shapes

::*
source	gradients*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
č
Rgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1W^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: 
ł
^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity
­
agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnterRefEnter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

bgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPush	StackPushagradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnterencoder_1/rnn/while/Identity^gradients/Add*
_output_shapes
:*
swap_memory( *Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
T0
Ą
jgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop/RefEnterRefEnter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
å
agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopStackPopjgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop/RefEnter^gradients/Sub*
	elem_type0*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
: 
©
Xgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Vgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/SliceRgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: *
T0
¤
gradients/AddN_7AddN2gradients/encoder_1/rnn/while/Select_2_grad/Select[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Slice_1*
T0*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_2_grad/Select*
N* 
_output_shapes
:


Bgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
_output_shapes
: *
dtype0*
valueB
 *    
¤
Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterBgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/

Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Jgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
N*
T0*
_output_shapes
: : 
Ė
Cgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*
_output_shapes
: : *
T0

@gradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddEgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Xgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
¾
Jgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration@gradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
_output_shapes
: *
T0
²
Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitCgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
_output_shapes
: *
T0

;gradients/encoder_1/rnn/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_7* 
_output_shapes
:
*
T0
Ų
ygradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3encoder_1/rnn/TensorArray_1Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
source	gradients*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes

::

ugradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3z^gradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *.
_class$
" loc:@encoder_1/rnn/TensorArray_1

kgradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3ygradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3&encoder_1/rnn/TensorArrayUnstack/rangeugradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*%
_output_shapes
:*
dtype0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
element_shape:

4gradients/encoder_1/transpose_grad/InvertPermutationInvertPermutationencoder_1/transpose/perm*
_output_shapes
:*
T0

,gradients/encoder_1/transpose_grad/transpose	Transposekgradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV34gradients/encoder_1/transpose_grad/InvertPermutation*
Tperm0*
T0*%
_output_shapes
:

+gradients/encoder/conv1d/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
Õ
-gradients/encoder/conv1d/Squeeze_grad/ReshapeReshape,gradients/encoder_1/transpose_grad/transpose+gradients/encoder/conv1d/Squeeze_grad/Shape*
T0*)
_output_shapes
:*
Tshape0

*gradients/encoder/conv1d/Conv2D_grad/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
Ņ
8gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/encoder/conv1d/Conv2D_grad/Shapeencoder/conv1d/ExpandDims_1-gradients/encoder/conv1d/Squeeze_grad/Reshape*
paddingVALID*
T0*
data_formatNHWC*
strides
*(
_output_shapes
:*
use_cudnn_on_gpu(

,gradients/encoder/conv1d/Conv2D_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            
Ó
9gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/conv1d/ExpandDims,gradients/encoder/conv1d/Conv2D_grad/Shape_1-gradients/encoder/conv1d/Squeeze_grad/Reshape*'
_output_shapes
:*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0

0gradients/encoder/conv1d/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
ę
2gradients/encoder/conv1d/ExpandDims_1_grad/ReshapeReshape9gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropFilter0gradients/encoder/conv1d/ExpandDims_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:
ø
global_norm/L2LossL2Loss2gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*
T0*
_output_shapes
: *E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape
ŗ
global_norm/L2Loss_1L2Loss2gradients/encoder_1/initial_state_0_tiled_grad/Sum*
T0*
_output_shapes
: *E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum
ŗ
global_norm/L2Loss_2L2Loss2gradients/encoder_1/initial_state_1_tiled_grad/Sum*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
_output_shapes
: 
ŗ
global_norm/L2Loss_3L2Loss2gradients/encoder_1/initial_state_2_tiled_grad/Sum*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum*
_output_shapes
: 
ŗ
global_norm/L2Loss_4L2Loss2gradients/encoder_1/initial_state_3_tiled_grad/Sum*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
_output_shapes
: *
T0

global_norm/L2Loss_5L2Lossagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*
_output_shapes
: *t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3

global_norm/L2Loss_6L2LossXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: *k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0

global_norm/L2Loss_7L2Lossagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0

global_norm/L2Loss_8L2LossXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: *
T0
Ź
global_norm/L2Loss_9L2Loss:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
_output_shapes
: *
T0
Ć
global_norm/L2Loss_10L2Loss6gradients/output_projection/xw_plus_b_grad/BiasAddGrad*I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
_output_shapes
: *
T0
Ä
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3global_norm/L2Loss_4global_norm/L2Loss_5global_norm/L2Loss_6global_norm/L2Loss_7global_norm/L2Loss_8global_norm/L2Loss_9global_norm/L2Loss_10*

axis *
_output_shapes
:*
T0*
N
[
global_norm/ConstConst*
valueB: *
_output_shapes
:*
dtype0
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
X
global_norm/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *   @
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
_output_shapes
: *
T0
Q
global_norm/global_normSqrtglobal_norm/mul*
T0*
_output_shapes
: 
b
clip_by_global_norm/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xglobal_norm/global_norm*
_output_shapes
: *
T0
^
clip_by_global_norm/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
d
clip_by_global_norm/truediv_1/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0

clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 

clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
_output_shapes
: *
T0
^
clip_by_global_norm/mul/xConst*
valueB
 *   @*
_output_shapes
: *
dtype0
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
T0*
_output_shapes
: 
ā
clip_by_global_norm/mul_1Mul2gradients/encoder/conv1d/ExpandDims_1_grad/Reshapeclip_by_global_norm/mul*
T0*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*#
_output_shapes
:
Ę
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*#
_output_shapes
:*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*
T0
Ž
clip_by_global_norm/mul_2Mul2gradients/encoder_1/initial_state_0_tiled_grad/Sumclip_by_global_norm/mul*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
:	
Ā
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
:	
Ž
clip_by_global_norm/mul_3Mul2gradients/encoder_1/initial_state_1_tiled_grad/Sumclip_by_global_norm/mul*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
_output_shapes
:	*
T0
Ā
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
_output_shapes
:	
Ž
clip_by_global_norm/mul_4Mul2gradients/encoder_1/initial_state_2_tiled_grad/Sumclip_by_global_norm/mul*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum*
_output_shapes
:	
Ā
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum*
_output_shapes
:	
Ž
clip_by_global_norm/mul_5Mul2gradients/encoder_1/initial_state_3_tiled_grad/Sumclip_by_global_norm/mul*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
_output_shapes
:	
Ā
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*
_output_shapes
:	*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
T0
½
clip_by_global_norm/mul_6Mulagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
*
T0
ņ
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*
T0* 
_output_shapes
:
*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3
¦
clip_by_global_norm/mul_7MulXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
_output_shapes	
:*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
ä
*clip_by_global_norm/clip_by_global_norm/_6Identityclip_by_global_norm/mul_7*
T0*
_output_shapes	
:*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3
½
clip_by_global_norm/mul_8Mulagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
*
T0
ņ
*clip_by_global_norm/clip_by_global_norm/_7Identityclip_by_global_norm/mul_8* 
_output_shapes
:
*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
¦
clip_by_global_norm/mul_9MulXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:*
T0
ä
*clip_by_global_norm/clip_by_global_norm/_8Identityclip_by_global_norm/mul_9*
_output_shapes	
:*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
ļ
clip_by_global_norm/mul_10Mul:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1clip_by_global_norm/mul*
_output_shapes
:	*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
T0
Ė
*clip_by_global_norm/clip_by_global_norm/_9Identityclip_by_global_norm/mul_10*
T0*
_output_shapes
:	*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1
ā
clip_by_global_norm/mul_11Mul6gradients/output_projection/xw_plus_b_grad/BiasAddGradclip_by_global_norm/mul*
_output_shapes
:*I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
T0
Ć
+clip_by_global_norm/clip_by_global_norm/_10Identityclip_by_global_norm/mul_11*I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
_output_shapes
:*
T0
p
grad_norms/grad_norms/tagsConst*&
valueB Bgrad_norms/grad_norms*
_output_shapes
: *
dtype0
|
grad_norms/grad_normsScalarSummarygrad_norms/grad_norms/tagsglobal_norm/global_norm*
T0*
_output_shapes
: 
~
beta1_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *fff?*
_class
loc:@input_embed

beta1_power
VariableV2*
_class
loc:@input_embed*
_output_shapes
: *
shape: *
dtype0*
shared_name *
	container 
®
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@input_embed
j
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
_class
loc:@input_embed*
T0
~
beta2_power/initial_valueConst*
valueB
 *w¾?*
_class
loc:@input_embed*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
	container *
dtype0*
_class
loc:@input_embed*
_output_shapes
: *
shape: *
shared_name 
®
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@input_embed*
validate_shape(*
_output_shapes
: 
j
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@input_embed*
_output_shapes
: 
d
zerosConst*"
valueB*    *#
_output_shapes
:*
dtype0
®
input_embed/Adam
VariableV2*
shared_name *
_class
loc:@input_embed*
	container *
shape:*
dtype0*#
_output_shapes
:
±
input_embed/Adam/AssignAssigninput_embed/Adamzeros*
_class
loc:@input_embed*#
_output_shapes
:*
T0*
validate_shape(*
use_locking(

input_embed/Adam/readIdentityinput_embed/Adam*
T0*
_class
loc:@input_embed*#
_output_shapes
:
f
zeros_1Const*"
valueB*    *#
_output_shapes
:*
dtype0
°
input_embed/Adam_1
VariableV2*
shape:*#
_output_shapes
:*
shared_name *
_class
loc:@input_embed*
dtype0*
	container 
·
input_embed/Adam_1/AssignAssigninput_embed/Adam_1zeros_1*
_class
loc:@input_embed*#
_output_shapes
:*
T0*
validate_shape(*
use_locking(

input_embed/Adam_1/readIdentityinput_embed/Adam_1*
T0*
_class
loc:@input_embed*#
_output_shapes
:
^
zeros_2Const*
valueB	*    *
dtype0*
_output_shapes
:	
¾
encoder/initial_state_0/Adam
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
shape:	*
shared_name 
Ó
#encoder/initial_state_0/Adam/AssignAssignencoder/initial_state_0/Adamzeros_2*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(
”
!encoder/initial_state_0/Adam/readIdentityencoder/initial_state_0/Adam*
T0**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	
^
zeros_3Const*
valueB	*    *
_output_shapes
:	*
dtype0
Ą
encoder/initial_state_0/Adam_1
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
shape:	*
shared_name 
×
%encoder/initial_state_0/Adam_1/AssignAssignencoder/initial_state_0/Adam_1zeros_3**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
„
#encoder/initial_state_0/Adam_1/readIdentityencoder/initial_state_0/Adam_1*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0*
T0
^
zeros_4Const*
valueB	*    *
_output_shapes
:	*
dtype0
¾
encoder/initial_state_1/Adam
VariableV2*
shared_name **
_class 
loc:@encoder/initial_state_1*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ó
#encoder/initial_state_1/Adam/AssignAssignencoder/initial_state_1/Adamzeros_4*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_1*
T0*
use_locking(
”
!encoder/initial_state_1/Adam/readIdentityencoder/initial_state_1/Adam*
T0**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	
^
zeros_5Const*
valueB	*    *
dtype0*
_output_shapes
:	
Ą
encoder/initial_state_1/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
shape:	*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1
×
%encoder/initial_state_1/Adam_1/AssignAssignencoder/initial_state_1/Adam_1zeros_5**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
„
#encoder/initial_state_1/Adam_1/readIdentityencoder/initial_state_1/Adam_1*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1
^
zeros_6Const*
valueB	*    *
_output_shapes
:	*
dtype0
¾
encoder/initial_state_2/Adam
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_2*
shared_name *
_output_shapes
:	*
shape:	
Ó
#encoder/initial_state_2/Adam/AssignAssignencoder/initial_state_2/Adamzeros_6*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2
”
!encoder/initial_state_2/Adam/readIdentityencoder/initial_state_2/Adam*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2
^
zeros_7Const*
valueB	*    *
_output_shapes
:	*
dtype0
Ą
encoder/initial_state_2/Adam_1
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_2*
shared_name *
_output_shapes
:	*
shape:	
×
%encoder/initial_state_2/Adam_1/AssignAssignencoder/initial_state_2/Adam_1zeros_7*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_2*
T0*
use_locking(
„
#encoder/initial_state_2/Adam_1/readIdentityencoder/initial_state_2/Adam_1**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	*
T0
^
zeros_8Const*
valueB	*    *
_output_shapes
:	*
dtype0
¾
encoder/initial_state_3/Adam
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_3*
shared_name *
_output_shapes
:	*
shape:	
Ó
#encoder/initial_state_3/Adam/AssignAssignencoder/initial_state_3/Adamzeros_8**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
”
!encoder/initial_state_3/Adam/readIdentityencoder/initial_state_3/Adam*
T0**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	
^
zeros_9Const*
_output_shapes
:	*
dtype0*
valueB	*    
Ą
encoder/initial_state_3/Adam_1
VariableV2*
shape:	*
_output_shapes
:	*
shared_name **
_class 
loc:@encoder/initial_state_3*
dtype0*
	container 
×
%encoder/initial_state_3/Adam_1/AssignAssignencoder/initial_state_3/Adam_1zeros_9*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3
„
#encoder/initial_state_3/Adam_1/readIdentityencoder/initial_state_3/Adam_1*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3
a
zeros_10Const*
dtype0* 
_output_shapes
:
*
valueB
*    
ų
8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam
VariableV2*
	container *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
shape:
*
shared_name 
©
?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/AssignAssign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamzeros_10*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
validate_shape(* 
_output_shapes
:

ö
=encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/readIdentity8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
a
zeros_11Const*
valueB
*    *
dtype0* 
_output_shapes
:

ś
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1
VariableV2*
	container *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
shape:
*
shared_name 
­
Aencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/AssignAssign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1zeros_11*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
ś
?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/readIdentity:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
W
zeros_12Const*
valueB*    *
dtype0*
_output_shapes	
:
ģ
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
”
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adamzeros_12*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
ī
<encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/readIdentity7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0
W
zeros_13Const*
valueB*    *
_output_shapes	
:*
dtype0
ī
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1
VariableV2*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
shape:*
dtype0*
shared_name *
	container 
„
@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1zeros_13*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
validate_shape(*
_output_shapes	
:
ņ
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/readIdentity9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
T0
a
zeros_14Const*
valueB
*    * 
_output_shapes
:
*
dtype0
ų
8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam
VariableV2*
shared_name *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
	container *
shape:
*
dtype0* 
_output_shapes
:

©
?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/AssignAssign8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adamzeros_14*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
ö
=encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/readIdentity8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0
a
zeros_15Const*
dtype0* 
_output_shapes
:
*
valueB
*    
ś
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1
VariableV2*
shared_name *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
	container *
shape:
*
dtype0* 
_output_shapes
:

­
Aencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/AssignAssign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1zeros_15*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
validate_shape(* 
_output_shapes
:

ś
?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/readIdentity:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
W
zeros_16Const*
valueB*    *
_output_shapes	
:*
dtype0
ģ
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam
VariableV2*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:*
shape:*
dtype0*
shared_name *
	container 
”
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adamzeros_16*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
ī
<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/readIdentity7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
W
zeros_17Const*
_output_shapes	
:*
dtype0*
valueB*    
ī
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1
VariableV2*
shape:*
_output_shapes	
:*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
dtype0*
	container 
„
@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1zeros_17*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
ņ
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/readIdentity9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
_
zeros_18Const*
_output_shapes
:	*
dtype0*
valueB	*    
¶
output_projection/W/Adam
VariableV2*
shape:	*
_output_shapes
:	*
shared_name *&
_class
loc:@output_projection/W*
dtype0*
	container 
Č
output_projection/W/Adam/AssignAssignoutput_projection/W/Adamzeros_18*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

output_projection/W/Adam/readIdentityoutput_projection/W/Adam*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0
_
zeros_19Const*
valueB	*    *
dtype0*
_output_shapes
:	
ø
output_projection/W/Adam_1
VariableV2*
shared_name *&
_class
loc:@output_projection/W*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ģ
!output_projection/W/Adam_1/AssignAssignoutput_projection/W/Adam_1zeros_19*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

output_projection/W/Adam_1/readIdentityoutput_projection/W/Adam_1*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0
U
zeros_20Const*
valueB*    *
_output_shapes
:*
dtype0
¬
output_projection/b/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_output_shapes
:*&
_class
loc:@output_projection/b
Ć
output_projection/b/Adam/AssignAssignoutput_projection/b/Adamzeros_20*
use_locking(*
T0*&
_class
loc:@output_projection/b*
validate_shape(*
_output_shapes
:

output_projection/b/Adam/readIdentityoutput_projection/b/Adam*
_output_shapes
:*&
_class
loc:@output_projection/b*
T0
U
zeros_21Const*
dtype0*
_output_shapes
:*
valueB*    
®
output_projection/b/Adam_1
VariableV2*
	container *
dtype0*&
_class
loc:@output_projection/b*
_output_shapes
:*
shape:*
shared_name 
Ē
!output_projection/b/Adam_1/AssignAssignoutput_projection/b/Adam_1zeros_21*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

output_projection/b/Adam_1/readIdentityoutput_projection/b/Adam_1*
_output_shapes
:*&
_class
loc:@output_projection/b*
T0
O

Adam/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
O

Adam/beta2Const*
valueB
 *w¾?*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wĢ+2
Ē
!Adam/update_input_embed/ApplyAdam	ApplyAdaminput_embedinput_embed/Adaminput_embed/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
use_locking( *
T0*
_class
loc:@input_embed*#
_output_shapes
:
’
-Adam/update_encoder/initial_state_0/ApplyAdam	ApplyAdamencoder/initial_state_0encoder/initial_state_0/Adamencoder/initial_state_0/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_locking( *
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0
’
-Adam/update_encoder/initial_state_1/ApplyAdam	ApplyAdamencoder/initial_state_1encoder/initial_state_1/Adamencoder/initial_state_1/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
use_locking( *
T0**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	
’
-Adam/update_encoder/initial_state_2/ApplyAdam	ApplyAdamencoder/initial_state_2encoder/initial_state_2/Adamencoder/initial_state_2/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
use_locking( *
T0**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	
’
-Adam/update_encoder/initial_state_3/ApplyAdam	ApplyAdamencoder/initial_state_3encoder/initial_state_3/Adamencoder/initial_state_3/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_4**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	*
T0*
use_locking( 

IAdam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdam	ApplyAdam3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_5*
use_locking( *
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights

HAdam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdam	ApplyAdam2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_6*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0*
use_locking( 

IAdam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdam	ApplyAdam3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_7*
use_locking( *
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:


HAdam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam	ApplyAdam2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_8*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
use_locking( 
ė
)Adam/update_output_projection/W/ApplyAdam	ApplyAdamoutput_projection/Woutput_projection/W/Adamoutput_projection/W/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_9*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0*
use_locking( 
ē
)Adam/update_output_projection/b/ApplyAdam	ApplyAdamoutput_projection/boutput_projection/b/Adamoutput_projection/b/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_10*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0*
use_locking( 
Ų
Adam/mulMulbeta1_power/read
Adam/beta1"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam*
_class
loc:@input_embed*
_output_shapes
: *
T0

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@input_embed
Ś

Adam/mul_1Mulbeta2_power/read
Adam/beta2"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam*
_output_shapes
: *
_class
loc:@input_embed*
T0

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@input_embed

Adam/updateNoOp"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam^Adam/Assign^Adam/Assign_1
w

Adam/valueConst^Adam/update*
dtype0*
_output_shapes
: *
value	B :*
_class
loc:@Variable
x
Adam	AssignAddVariable
Adam/value*
use_locking( *
T0*
_class
loc:@Variable*
_output_shapes
: 
Ö
Const_1Const*
_output_shapes	
:*
dtype0	*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
R
ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
b
ArgMaxArgMaxcond/Merge_1ArgMax/dimension*

Tidx0*
T0*
_output_shapes	
:
E
EqualEqualArgMaxConst_1*
T0	*
_output_shapes	
:
L

LogicalAnd
LogicalAndaccuracy/EqualEqual*
_output_shapes	
:
Z
count_nonzero/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
n
count_nonzero/NotEqualNotEqual
LogicalAndcount_nonzero/NotEqual/y*
_output_shapes	
:*
T0

j
count_nonzero/ToInt64Castcount_nonzero/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
]
count_nonzero/ConstConst*
valueB: *
dtype0*
_output_shapes
:

count_nonzero/SumSumcount_nonzero/ToInt64count_nonzero/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
Y
Equal_1Equaloutput_projection/ArgMaxConst_1*
T0	*
_output_shapes	
:
\
count_nonzero_1/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

o
count_nonzero_1/NotEqualNotEqualEqual_1count_nonzero_1/NotEqual/y*
T0
*
_output_shapes	
:
n
count_nonzero_1/ToInt64Castcount_nonzero_1/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
_
count_nonzero_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

count_nonzero_1/SumSumcount_nonzero_1/ToInt64count_nonzero_1/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
f
ArgMax_1ArgMaxcond/Merge_1ArgMax_1/dimension*
_output_shapes	
:*
T0*

Tidx0
I
Equal_2EqualArgMax_1Const_1*
_output_shapes	
:*
T0	
\
count_nonzero_2/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
o
count_nonzero_2/NotEqualNotEqualEqual_2count_nonzero_2/NotEqual/y*
T0
*
_output_shapes	
:
n
count_nonzero_2/ToInt64Castcount_nonzero_2/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

_
count_nonzero_2/ConstConst*
valueB: *
_output_shapes
:*
dtype0

count_nonzero_2/SumSumcount_nonzero_2/ToInt64count_nonzero_2/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
K
	Greater/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
S
GreaterGreatercount_nonzero_1/Sum	Greater/y*
_output_shapes
: *
T0	
L
cond_1/SwitchSwitchGreaterGreater*
_output_shapes
: : *
T0

M
cond_1/switch_tIdentitycond_1/Switch:1*
_output_shapes
: *
T0

K
cond_1/switch_fIdentitycond_1/Switch*
T0
*
_output_shapes
: 
D
cond_1/pred_idIdentityGreater*
T0
*
_output_shapes
: 

cond_1/truediv/Cast/SwitchSwitchcount_nonzero/Sumcond_1/pred_id*
_output_shapes
: : *$
_class
loc:@count_nonzero/Sum*
T0	
i
cond_1/truediv/CastCastcond_1/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_1/truediv/Cast_1/SwitchSwitchcount_nonzero_1/Sumcond_1/pred_id*
_output_shapes
: : *&
_class
loc:@count_nonzero_1/Sum*
T0	
m
cond_1/truediv/Cast_1Castcond_1/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
f
cond_1/truedivRealDivcond_1/truediv/Castcond_1/truediv/Cast_1*
_output_shapes
: *
T0
g
cond_1/ConstConst^cond_1/switch_f*
valueB 2        *
_output_shapes
: *
dtype0
_
cond_1/MergeMergecond_1/Constcond_1/truediv*
_output_shapes
: : *
T0*
N
\
precision_0/tagsConst*
valueB Bprecision_0*
_output_shapes
: *
dtype0
]
precision_0ScalarSummaryprecision_0/tagscond_1/Merge*
_output_shapes
: *
T0
M
Greater_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
W
	Greater_1Greatercount_nonzero_2/SumGreater_1/y*
_output_shapes
: *
T0	
P
cond_2/SwitchSwitch	Greater_1	Greater_1*
_output_shapes
: : *
T0

M
cond_2/switch_tIdentitycond_2/Switch:1*
_output_shapes
: *
T0

K
cond_2/switch_fIdentitycond_2/Switch*
T0
*
_output_shapes
: 
F
cond_2/pred_idIdentity	Greater_1*
T0
*
_output_shapes
: 

cond_2/truediv/Cast/SwitchSwitchcount_nonzero/Sumcond_2/pred_id*
T0	*
_output_shapes
: : *$
_class
loc:@count_nonzero/Sum
i
cond_2/truediv/CastCastcond_2/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	

cond_2/truediv/Cast_1/SwitchSwitchcount_nonzero_2/Sumcond_2/pred_id*&
_class
loc:@count_nonzero_2/Sum*
_output_shapes
: : *
T0	
m
cond_2/truediv/Cast_1Castcond_2/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
f
cond_2/truedivRealDivcond_2/truediv/Castcond_2/truediv/Cast_1*
_output_shapes
: *
T0
g
cond_2/ConstConst^cond_2/switch_f*
valueB 2        *
_output_shapes
: *
dtype0
_
cond_2/MergeMergecond_2/Constcond_2/truediv*
_output_shapes
: : *
N*
T0
V
recall_0/tagsConst*
valueB Brecall_0*
dtype0*
_output_shapes
: 
W
recall_0ScalarSummaryrecall_0/tagscond_2/Merge*
_output_shapes
: *
T0
Ö
Const_2Const*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                *
dtype0	*
_output_shapes	
:
T
ArgMax_2/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
f
ArgMax_2ArgMaxcond/Merge_1ArgMax_2/dimension*

Tidx0*
T0*
_output_shapes	
:
I
Equal_3EqualArgMax_2Const_2*
T0	*
_output_shapes	
:
P
LogicalAnd_1
LogicalAndaccuracy/EqualEqual_3*
_output_shapes	
:
\
count_nonzero_3/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

t
count_nonzero_3/NotEqualNotEqualLogicalAnd_1count_nonzero_3/NotEqual/y*
T0
*
_output_shapes	
:
n
count_nonzero_3/ToInt64Castcount_nonzero_3/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

_
count_nonzero_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

count_nonzero_3/SumSumcount_nonzero_3/ToInt64count_nonzero_3/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
Y
Equal_4Equaloutput_projection/ArgMaxConst_2*
T0	*
_output_shapes	
:
\
count_nonzero_4/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

o
count_nonzero_4/NotEqualNotEqualEqual_4count_nonzero_4/NotEqual/y*
_output_shapes	
:*
T0

n
count_nonzero_4/ToInt64Castcount_nonzero_4/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

_
count_nonzero_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

count_nonzero_4/SumSumcount_nonzero_4/ToInt64count_nonzero_4/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
T
ArgMax_3/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
f
ArgMax_3ArgMaxcond/Merge_1ArgMax_3/dimension*
_output_shapes	
:*
T0*

Tidx0
I
Equal_5EqualArgMax_3Const_2*
T0	*
_output_shapes	
:
\
count_nonzero_5/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
o
count_nonzero_5/NotEqualNotEqualEqual_5count_nonzero_5/NotEqual/y*
T0
*
_output_shapes	
:
n
count_nonzero_5/ToInt64Castcount_nonzero_5/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

_
count_nonzero_5/ConstConst*
valueB: *
dtype0*
_output_shapes
:

count_nonzero_5/SumSumcount_nonzero_5/ToInt64count_nonzero_5/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
M
Greater_2/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
W
	Greater_2Greatercount_nonzero_4/SumGreater_2/y*
T0	*
_output_shapes
: 
P
cond_3/SwitchSwitch	Greater_2	Greater_2*
T0
*
_output_shapes
: : 
M
cond_3/switch_tIdentitycond_3/Switch:1*
_output_shapes
: *
T0

K
cond_3/switch_fIdentitycond_3/Switch*
_output_shapes
: *
T0

F
cond_3/pred_idIdentity	Greater_2*
T0
*
_output_shapes
: 

cond_3/truediv/Cast/SwitchSwitchcount_nonzero_3/Sumcond_3/pred_id*
T0	*
_output_shapes
: : *&
_class
loc:@count_nonzero_3/Sum
i
cond_3/truediv/CastCastcond_3/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	

cond_3/truediv/Cast_1/SwitchSwitchcount_nonzero_4/Sumcond_3/pred_id*&
_class
loc:@count_nonzero_4/Sum*
_output_shapes
: : *
T0	
m
cond_3/truediv/Cast_1Castcond_3/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
f
cond_3/truedivRealDivcond_3/truediv/Castcond_3/truediv/Cast_1*
_output_shapes
: *
T0
g
cond_3/ConstConst^cond_3/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
_
cond_3/MergeMergecond_3/Constcond_3/truediv*
T0*
N*
_output_shapes
: : 
\
precision_1/tagsConst*
valueB Bprecision_1*
dtype0*
_output_shapes
: 
]
precision_1ScalarSummaryprecision_1/tagscond_3/Merge*
_output_shapes
: *
T0
M
Greater_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
W
	Greater_3Greatercount_nonzero_5/SumGreater_3/y*
_output_shapes
: *
T0	
P
cond_4/SwitchSwitch	Greater_3	Greater_3*
T0
*
_output_shapes
: : 
M
cond_4/switch_tIdentitycond_4/Switch:1*
T0
*
_output_shapes
: 
K
cond_4/switch_fIdentitycond_4/Switch*
T0
*
_output_shapes
: 
F
cond_4/pred_idIdentity	Greater_3*
_output_shapes
: *
T0


cond_4/truediv/Cast/SwitchSwitchcount_nonzero_3/Sumcond_4/pred_id*
T0	*
_output_shapes
: : *&
_class
loc:@count_nonzero_3/Sum
i
cond_4/truediv/CastCastcond_4/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	

cond_4/truediv/Cast_1/SwitchSwitchcount_nonzero_5/Sumcond_4/pred_id*&
_class
loc:@count_nonzero_5/Sum*
_output_shapes
: : *
T0	
m
cond_4/truediv/Cast_1Castcond_4/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
f
cond_4/truedivRealDivcond_4/truediv/Castcond_4/truediv/Cast_1*
T0*
_output_shapes
: 
g
cond_4/ConstConst^cond_4/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
_
cond_4/MergeMergecond_4/Constcond_4/truediv*
T0*
N*
_output_shapes
: : 
V
recall_1/tagsConst*
valueB Brecall_1*
dtype0*
_output_shapes
: 
W
recall_1ScalarSummaryrecall_1/tagscond_4/Merge*
_output_shapes
: *
T0
Ö
Const_3Const*
dtype0	*
_output_shapes	
:*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
T
ArgMax_4/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
f
ArgMax_4ArgMaxcond/Merge_1ArgMax_4/dimension*
_output_shapes	
:*
T0*

Tidx0
I
Equal_6EqualArgMax_4Const_3*
T0	*
_output_shapes	
:
P
LogicalAnd_2
LogicalAndaccuracy/EqualEqual_6*
_output_shapes	
:
\
count_nonzero_6/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

t
count_nonzero_6/NotEqualNotEqualLogicalAnd_2count_nonzero_6/NotEqual/y*
T0
*
_output_shapes	
:
n
count_nonzero_6/ToInt64Castcount_nonzero_6/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

_
count_nonzero_6/ConstConst*
valueB: *
_output_shapes
:*
dtype0

count_nonzero_6/SumSumcount_nonzero_6/ToInt64count_nonzero_6/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
Y
Equal_7Equaloutput_projection/ArgMaxConst_3*
T0	*
_output_shapes	
:
\
count_nonzero_7/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

o
count_nonzero_7/NotEqualNotEqualEqual_7count_nonzero_7/NotEqual/y*
T0
*
_output_shapes	
:
n
count_nonzero_7/ToInt64Castcount_nonzero_7/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

_
count_nonzero_7/ConstConst*
valueB: *
dtype0*
_output_shapes
:

count_nonzero_7/SumSumcount_nonzero_7/ToInt64count_nonzero_7/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
T
ArgMax_5/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
f
ArgMax_5ArgMaxcond/Merge_1ArgMax_5/dimension*
_output_shapes	
:*
T0*

Tidx0
I
Equal_8EqualArgMax_5Const_3*
_output_shapes	
:*
T0	
\
count_nonzero_8/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
o
count_nonzero_8/NotEqualNotEqualEqual_8count_nonzero_8/NotEqual/y*
T0
*
_output_shapes	
:
n
count_nonzero_8/ToInt64Castcount_nonzero_8/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

_
count_nonzero_8/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

count_nonzero_8/SumSumcount_nonzero_8/ToInt64count_nonzero_8/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
M
Greater_4/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
W
	Greater_4Greatercount_nonzero_7/SumGreater_4/y*
T0	*
_output_shapes
: 
P
cond_5/SwitchSwitch	Greater_4	Greater_4*
T0
*
_output_shapes
: : 
M
cond_5/switch_tIdentitycond_5/Switch:1*
_output_shapes
: *
T0

K
cond_5/switch_fIdentitycond_5/Switch*
_output_shapes
: *
T0

F
cond_5/pred_idIdentity	Greater_4*
T0
*
_output_shapes
: 

cond_5/truediv/Cast/SwitchSwitchcount_nonzero_6/Sumcond_5/pred_id*
T0	*
_output_shapes
: : *&
_class
loc:@count_nonzero_6/Sum
i
cond_5/truediv/CastCastcond_5/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_5/truediv/Cast_1/SwitchSwitchcount_nonzero_7/Sumcond_5/pred_id*
T0	*&
_class
loc:@count_nonzero_7/Sum*
_output_shapes
: : 
m
cond_5/truediv/Cast_1Castcond_5/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
f
cond_5/truedivRealDivcond_5/truediv/Castcond_5/truediv/Cast_1*
_output_shapes
: *
T0
g
cond_5/ConstConst^cond_5/switch_f*
valueB 2        *
dtype0*
_output_shapes
: 
_
cond_5/MergeMergecond_5/Constcond_5/truediv*
_output_shapes
: : *
T0*
N
\
precision_2/tagsConst*
dtype0*
_output_shapes
: *
valueB Bprecision_2
]
precision_2ScalarSummaryprecision_2/tagscond_5/Merge*
_output_shapes
: *
T0
M
Greater_5/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
W
	Greater_5Greatercount_nonzero_8/SumGreater_5/y*
T0	*
_output_shapes
: 
P
cond_6/SwitchSwitch	Greater_5	Greater_5*
_output_shapes
: : *
T0

M
cond_6/switch_tIdentitycond_6/Switch:1*
T0
*
_output_shapes
: 
K
cond_6/switch_fIdentitycond_6/Switch*
T0
*
_output_shapes
: 
F
cond_6/pred_idIdentity	Greater_5*
_output_shapes
: *
T0


cond_6/truediv/Cast/SwitchSwitchcount_nonzero_6/Sumcond_6/pred_id*
T0	*
_output_shapes
: : *&
_class
loc:@count_nonzero_6/Sum
i
cond_6/truediv/CastCastcond_6/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_6/truediv/Cast_1/SwitchSwitchcount_nonzero_8/Sumcond_6/pred_id*
T0	*&
_class
loc:@count_nonzero_8/Sum*
_output_shapes
: : 
m
cond_6/truediv/Cast_1Castcond_6/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
f
cond_6/truedivRealDivcond_6/truediv/Castcond_6/truediv/Cast_1*
T0*
_output_shapes
: 
g
cond_6/ConstConst^cond_6/switch_f*
dtype0*
_output_shapes
: *
valueB 2        
_
cond_6/MergeMergecond_6/Constcond_6/truediv*
_output_shapes
: : *
T0*
N
V
recall_2/tagsConst*
_output_shapes
: *
dtype0*
valueB Brecall_2
W
recall_2ScalarSummaryrecall_2/tagscond_6/Merge*
T0*
_output_shapes
: 
Ö
Const_4Const*
dtype0	*
_output_shapes	
:*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
T
ArgMax_6/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
f
ArgMax_6ArgMaxcond/Merge_1ArgMax_6/dimension*

Tidx0*
T0*
_output_shapes	
:
I
Equal_9EqualArgMax_6Const_4*
T0	*
_output_shapes	
:
P
LogicalAnd_3
LogicalAndaccuracy/EqualEqual_9*
_output_shapes	
:
\
count_nonzero_9/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
t
count_nonzero_9/NotEqualNotEqualLogicalAnd_3count_nonzero_9/NotEqual/y*
T0
*
_output_shapes	
:
n
count_nonzero_9/ToInt64Castcount_nonzero_9/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

_
count_nonzero_9/ConstConst*
valueB: *
dtype0*
_output_shapes
:

count_nonzero_9/SumSumcount_nonzero_9/ToInt64count_nonzero_9/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
Z
Equal_10Equaloutput_projection/ArgMaxConst_4*
T0	*
_output_shapes	
:
]
count_nonzero_10/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
r
count_nonzero_10/NotEqualNotEqualEqual_10count_nonzero_10/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_10/ToInt64Castcount_nonzero_10/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

`
count_nonzero_10/ConstConst*
valueB: *
_output_shapes
:*
dtype0

count_nonzero_10/SumSumcount_nonzero_10/ToInt64count_nonzero_10/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
T
ArgMax_7/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
f
ArgMax_7ArgMaxcond/Merge_1ArgMax_7/dimension*
_output_shapes	
:*
T0*

Tidx0
J
Equal_11EqualArgMax_7Const_4*
T0	*
_output_shapes	
:
]
count_nonzero_11/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
r
count_nonzero_11/NotEqualNotEqualEqual_11count_nonzero_11/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_11/ToInt64Castcount_nonzero_11/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
`
count_nonzero_11/ConstConst*
valueB: *
_output_shapes
:*
dtype0

count_nonzero_11/SumSumcount_nonzero_11/ToInt64count_nonzero_11/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
M
Greater_6/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
X
	Greater_6Greatercount_nonzero_10/SumGreater_6/y*
_output_shapes
: *
T0	
P
cond_7/SwitchSwitch	Greater_6	Greater_6*
T0
*
_output_shapes
: : 
M
cond_7/switch_tIdentitycond_7/Switch:1*
T0
*
_output_shapes
: 
K
cond_7/switch_fIdentitycond_7/Switch*
_output_shapes
: *
T0

F
cond_7/pred_idIdentity	Greater_6*
_output_shapes
: *
T0


cond_7/truediv/Cast/SwitchSwitchcount_nonzero_9/Sumcond_7/pred_id*
T0	*&
_class
loc:@count_nonzero_9/Sum*
_output_shapes
: : 
i
cond_7/truediv/CastCastcond_7/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_7/truediv/Cast_1/SwitchSwitchcount_nonzero_10/Sumcond_7/pred_id*'
_class
loc:@count_nonzero_10/Sum*
_output_shapes
: : *
T0	
m
cond_7/truediv/Cast_1Castcond_7/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
f
cond_7/truedivRealDivcond_7/truediv/Castcond_7/truediv/Cast_1*
_output_shapes
: *
T0
g
cond_7/ConstConst^cond_7/switch_f*
valueB 2        *
dtype0*
_output_shapes
: 
_
cond_7/MergeMergecond_7/Constcond_7/truediv*
_output_shapes
: : *
N*
T0
\
precision_3/tagsConst*
valueB Bprecision_3*
_output_shapes
: *
dtype0
]
precision_3ScalarSummaryprecision_3/tagscond_7/Merge*
T0*
_output_shapes
: 
M
Greater_7/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
X
	Greater_7Greatercount_nonzero_11/SumGreater_7/y*
_output_shapes
: *
T0	
P
cond_8/SwitchSwitch	Greater_7	Greater_7*
_output_shapes
: : *
T0

M
cond_8/switch_tIdentitycond_8/Switch:1*
T0
*
_output_shapes
: 
K
cond_8/switch_fIdentitycond_8/Switch*
T0
*
_output_shapes
: 
F
cond_8/pred_idIdentity	Greater_7*
_output_shapes
: *
T0


cond_8/truediv/Cast/SwitchSwitchcount_nonzero_9/Sumcond_8/pred_id*
T0	*&
_class
loc:@count_nonzero_9/Sum*
_output_shapes
: : 
i
cond_8/truediv/CastCastcond_8/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_8/truediv/Cast_1/SwitchSwitchcount_nonzero_11/Sumcond_8/pred_id*
T0	*
_output_shapes
: : *'
_class
loc:@count_nonzero_11/Sum
m
cond_8/truediv/Cast_1Castcond_8/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
f
cond_8/truedivRealDivcond_8/truediv/Castcond_8/truediv/Cast_1*
T0*
_output_shapes
: 
g
cond_8/ConstConst^cond_8/switch_f*
dtype0*
_output_shapes
: *
valueB 2        
_
cond_8/MergeMergecond_8/Constcond_8/truediv*
_output_shapes
: : *
T0*
N
V
recall_3/tagsConst*
valueB Brecall_3*
dtype0*
_output_shapes
: 
W
recall_3ScalarSummaryrecall_3/tagscond_8/Merge*
T0*
_output_shapes
: 
Ö
Const_5Const*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                *
_output_shapes	
:*
dtype0	
T
ArgMax_8/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
f
ArgMax_8ArgMaxcond/Merge_1ArgMax_8/dimension*
_output_shapes	
:*
T0*

Tidx0
J
Equal_12EqualArgMax_8Const_5*
T0	*
_output_shapes	
:
Q
LogicalAnd_4
LogicalAndaccuracy/EqualEqual_12*
_output_shapes	
:
]
count_nonzero_12/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
v
count_nonzero_12/NotEqualNotEqualLogicalAnd_4count_nonzero_12/NotEqual/y*
_output_shapes	
:*
T0

p
count_nonzero_12/ToInt64Castcount_nonzero_12/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
`
count_nonzero_12/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

count_nonzero_12/SumSumcount_nonzero_12/ToInt64count_nonzero_12/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
Z
Equal_13Equaloutput_projection/ArgMaxConst_5*
_output_shapes	
:*
T0	
]
count_nonzero_13/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
r
count_nonzero_13/NotEqualNotEqualEqual_13count_nonzero_13/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_13/ToInt64Castcount_nonzero_13/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
`
count_nonzero_13/ConstConst*
valueB: *
dtype0*
_output_shapes
:

count_nonzero_13/SumSumcount_nonzero_13/ToInt64count_nonzero_13/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
T
ArgMax_9/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
f
ArgMax_9ArgMaxcond/Merge_1ArgMax_9/dimension*
_output_shapes	
:*
T0*

Tidx0
J
Equal_14EqualArgMax_9Const_5*
T0	*
_output_shapes	
:
]
count_nonzero_14/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

r
count_nonzero_14/NotEqualNotEqualEqual_14count_nonzero_14/NotEqual/y*
_output_shapes	
:*
T0

p
count_nonzero_14/ToInt64Castcount_nonzero_14/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
`
count_nonzero_14/ConstConst*
valueB: *
dtype0*
_output_shapes
:

count_nonzero_14/SumSumcount_nonzero_14/ToInt64count_nonzero_14/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
M
Greater_8/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
X
	Greater_8Greatercount_nonzero_13/SumGreater_8/y*
T0	*
_output_shapes
: 
P
cond_9/SwitchSwitch	Greater_8	Greater_8*
_output_shapes
: : *
T0

M
cond_9/switch_tIdentitycond_9/Switch:1*
_output_shapes
: *
T0

K
cond_9/switch_fIdentitycond_9/Switch*
_output_shapes
: *
T0

F
cond_9/pred_idIdentity	Greater_8*
T0
*
_output_shapes
: 

cond_9/truediv/Cast/SwitchSwitchcount_nonzero_12/Sumcond_9/pred_id*
T0	*
_output_shapes
: : *'
_class
loc:@count_nonzero_12/Sum
i
cond_9/truediv/CastCastcond_9/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	

cond_9/truediv/Cast_1/SwitchSwitchcount_nonzero_13/Sumcond_9/pred_id*
T0	*
_output_shapes
: : *'
_class
loc:@count_nonzero_13/Sum
m
cond_9/truediv/Cast_1Castcond_9/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
f
cond_9/truedivRealDivcond_9/truediv/Castcond_9/truediv/Cast_1*
T0*
_output_shapes
: 
g
cond_9/ConstConst^cond_9/switch_f*
dtype0*
_output_shapes
: *
valueB 2        
_
cond_9/MergeMergecond_9/Constcond_9/truediv*
_output_shapes
: : *
T0*
N
\
precision_4/tagsConst*
_output_shapes
: *
dtype0*
valueB Bprecision_4
]
precision_4ScalarSummaryprecision_4/tagscond_9/Merge*
_output_shapes
: *
T0
M
Greater_9/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
X
	Greater_9Greatercount_nonzero_14/SumGreater_9/y*
_output_shapes
: *
T0	
Q
cond_10/SwitchSwitch	Greater_9	Greater_9*
T0
*
_output_shapes
: : 
O
cond_10/switch_tIdentitycond_10/Switch:1*
T0
*
_output_shapes
: 
M
cond_10/switch_fIdentitycond_10/Switch*
T0
*
_output_shapes
: 
G
cond_10/pred_idIdentity	Greater_9*
T0
*
_output_shapes
: 

cond_10/truediv/Cast/SwitchSwitchcount_nonzero_12/Sumcond_10/pred_id*'
_class
loc:@count_nonzero_12/Sum*
_output_shapes
: : *
T0	
k
cond_10/truediv/CastCastcond_10/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_10/truediv/Cast_1/SwitchSwitchcount_nonzero_14/Sumcond_10/pred_id*
T0	*
_output_shapes
: : *'
_class
loc:@count_nonzero_14/Sum
o
cond_10/truediv/Cast_1Castcond_10/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
i
cond_10/truedivRealDivcond_10/truediv/Castcond_10/truediv/Cast_1*
T0*
_output_shapes
: 
i
cond_10/ConstConst^cond_10/switch_f*
valueB 2        *
_output_shapes
: *
dtype0
b
cond_10/MergeMergecond_10/Constcond_10/truediv*
N*
T0*
_output_shapes
: : 
V
recall_4/tagsConst*
dtype0*
_output_shapes
: *
valueB Brecall_4
X
recall_4ScalarSummaryrecall_4/tagscond_10/Merge*
_output_shapes
: *
T0
Ö
Const_6Const*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                *
_output_shapes	
:*
dtype0	
U
ArgMax_10/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
h
	ArgMax_10ArgMaxcond/Merge_1ArgMax_10/dimension*

Tidx0*
T0*
_output_shapes	
:
K
Equal_15Equal	ArgMax_10Const_6*
T0	*
_output_shapes	
:
Q
LogicalAnd_5
LogicalAndaccuracy/EqualEqual_15*
_output_shapes	
:
]
count_nonzero_15/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
v
count_nonzero_15/NotEqualNotEqualLogicalAnd_5count_nonzero_15/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_15/ToInt64Castcount_nonzero_15/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

`
count_nonzero_15/ConstConst*
valueB: *
_output_shapes
:*
dtype0

count_nonzero_15/SumSumcount_nonzero_15/ToInt64count_nonzero_15/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
Z
Equal_16Equaloutput_projection/ArgMaxConst_6*
T0	*
_output_shapes	
:
]
count_nonzero_16/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
r
count_nonzero_16/NotEqualNotEqualEqual_16count_nonzero_16/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_16/ToInt64Castcount_nonzero_16/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

`
count_nonzero_16/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

count_nonzero_16/SumSumcount_nonzero_16/ToInt64count_nonzero_16/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
U
ArgMax_11/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
h
	ArgMax_11ArgMaxcond/Merge_1ArgMax_11/dimension*
_output_shapes	
:*
T0*

Tidx0
K
Equal_17Equal	ArgMax_11Const_6*
_output_shapes	
:*
T0	
]
count_nonzero_17/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
r
count_nonzero_17/NotEqualNotEqualEqual_17count_nonzero_17/NotEqual/y*
_output_shapes	
:*
T0

p
count_nonzero_17/ToInt64Castcount_nonzero_17/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

`
count_nonzero_17/ConstConst*
valueB: *
_output_shapes
:*
dtype0

count_nonzero_17/SumSumcount_nonzero_17/ToInt64count_nonzero_17/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
N
Greater_10/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Z

Greater_10Greatercount_nonzero_16/SumGreater_10/y*
T0	*
_output_shapes
: 
S
cond_11/SwitchSwitch
Greater_10
Greater_10*
_output_shapes
: : *
T0

O
cond_11/switch_tIdentitycond_11/Switch:1*
_output_shapes
: *
T0

M
cond_11/switch_fIdentitycond_11/Switch*
_output_shapes
: *
T0

H
cond_11/pred_idIdentity
Greater_10*
T0
*
_output_shapes
: 

cond_11/truediv/Cast/SwitchSwitchcount_nonzero_15/Sumcond_11/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_15/Sum*
T0	
k
cond_11/truediv/CastCastcond_11/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_11/truediv/Cast_1/SwitchSwitchcount_nonzero_16/Sumcond_11/pred_id*
T0	*'
_class
loc:@count_nonzero_16/Sum*
_output_shapes
: : 
o
cond_11/truediv/Cast_1Castcond_11/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
i
cond_11/truedivRealDivcond_11/truediv/Castcond_11/truediv/Cast_1*
T0*
_output_shapes
: 
i
cond_11/ConstConst^cond_11/switch_f*
valueB 2        *
_output_shapes
: *
dtype0
b
cond_11/MergeMergecond_11/Constcond_11/truediv*
T0*
N*
_output_shapes
: : 
\
precision_5/tagsConst*
valueB Bprecision_5*
_output_shapes
: *
dtype0
^
precision_5ScalarSummaryprecision_5/tagscond_11/Merge*
T0*
_output_shapes
: 
N
Greater_11/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Z

Greater_11Greatercount_nonzero_17/SumGreater_11/y*
T0	*
_output_shapes
: 
S
cond_12/SwitchSwitch
Greater_11
Greater_11*
_output_shapes
: : *
T0

O
cond_12/switch_tIdentitycond_12/Switch:1*
T0
*
_output_shapes
: 
M
cond_12/switch_fIdentitycond_12/Switch*
T0
*
_output_shapes
: 
H
cond_12/pred_idIdentity
Greater_11*
_output_shapes
: *
T0


cond_12/truediv/Cast/SwitchSwitchcount_nonzero_15/Sumcond_12/pred_id*
T0	*'
_class
loc:@count_nonzero_15/Sum*
_output_shapes
: : 
k
cond_12/truediv/CastCastcond_12/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_12/truediv/Cast_1/SwitchSwitchcount_nonzero_17/Sumcond_12/pred_id*'
_class
loc:@count_nonzero_17/Sum*
_output_shapes
: : *
T0	
o
cond_12/truediv/Cast_1Castcond_12/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
i
cond_12/truedivRealDivcond_12/truediv/Castcond_12/truediv/Cast_1*
T0*
_output_shapes
: 
i
cond_12/ConstConst^cond_12/switch_f*
valueB 2        *
_output_shapes
: *
dtype0
b
cond_12/MergeMergecond_12/Constcond_12/truediv*
N*
T0*
_output_shapes
: : 
V
recall_5/tagsConst*
_output_shapes
: *
dtype0*
valueB Brecall_5
X
recall_5ScalarSummaryrecall_5/tagscond_12/Merge*
T0*
_output_shapes
: 
Ö
Const_7Const*
_output_shapes	
:*
dtype0	*
valueB	"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
U
ArgMax_12/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
h
	ArgMax_12ArgMaxcond/Merge_1ArgMax_12/dimension*
_output_shapes	
:*
T0*

Tidx0
K
Equal_18Equal	ArgMax_12Const_7*
_output_shapes	
:*
T0	
Q
LogicalAnd_6
LogicalAndaccuracy/EqualEqual_18*
_output_shapes	
:
]
count_nonzero_18/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
v
count_nonzero_18/NotEqualNotEqualLogicalAnd_6count_nonzero_18/NotEqual/y*
_output_shapes	
:*
T0

p
count_nonzero_18/ToInt64Castcount_nonzero_18/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

`
count_nonzero_18/ConstConst*
valueB: *
_output_shapes
:*
dtype0

count_nonzero_18/SumSumcount_nonzero_18/ToInt64count_nonzero_18/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
Z
Equal_19Equaloutput_projection/ArgMaxConst_7*
_output_shapes	
:*
T0	
]
count_nonzero_19/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

r
count_nonzero_19/NotEqualNotEqualEqual_19count_nonzero_19/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_19/ToInt64Castcount_nonzero_19/NotEqual*

SrcT0
*
_output_shapes	
:*

DstT0	
`
count_nonzero_19/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

count_nonzero_19/SumSumcount_nonzero_19/ToInt64count_nonzero_19/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
U
ArgMax_13/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
h
	ArgMax_13ArgMaxcond/Merge_1ArgMax_13/dimension*

Tidx0*
T0*
_output_shapes	
:
K
Equal_20Equal	ArgMax_13Const_7*
_output_shapes	
:*
T0	
]
count_nonzero_20/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
r
count_nonzero_20/NotEqualNotEqualEqual_20count_nonzero_20/NotEqual/y*
T0
*
_output_shapes	
:
p
count_nonzero_20/ToInt64Castcount_nonzero_20/NotEqual*
_output_shapes	
:*

DstT0	*

SrcT0

`
count_nonzero_20/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

count_nonzero_20/SumSumcount_nonzero_20/ToInt64count_nonzero_20/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
N
Greater_12/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
Z

Greater_12Greatercount_nonzero_19/SumGreater_12/y*
_output_shapes
: *
T0	
S
cond_13/SwitchSwitch
Greater_12
Greater_12*
_output_shapes
: : *
T0

O
cond_13/switch_tIdentitycond_13/Switch:1*
T0
*
_output_shapes
: 
M
cond_13/switch_fIdentitycond_13/Switch*
T0
*
_output_shapes
: 
H
cond_13/pred_idIdentity
Greater_12*
_output_shapes
: *
T0


cond_13/truediv/Cast/SwitchSwitchcount_nonzero_18/Sumcond_13/pred_id*
T0	*
_output_shapes
: : *'
_class
loc:@count_nonzero_18/Sum
k
cond_13/truediv/CastCastcond_13/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0

cond_13/truediv/Cast_1/SwitchSwitchcount_nonzero_19/Sumcond_13/pred_id*'
_class
loc:@count_nonzero_19/Sum*
_output_shapes
: : *
T0	
o
cond_13/truediv/Cast_1Castcond_13/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
i
cond_13/truedivRealDivcond_13/truediv/Castcond_13/truediv/Cast_1*
T0*
_output_shapes
: 
i
cond_13/ConstConst^cond_13/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
b
cond_13/MergeMergecond_13/Constcond_13/truediv*
T0*
N*
_output_shapes
: : 
\
precision_6/tagsConst*
_output_shapes
: *
dtype0*
valueB Bprecision_6
^
precision_6ScalarSummaryprecision_6/tagscond_13/Merge*
_output_shapes
: *
T0
N
Greater_13/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Z

Greater_13Greatercount_nonzero_20/SumGreater_13/y*
_output_shapes
: *
T0	
S
cond_14/SwitchSwitch
Greater_13
Greater_13*
_output_shapes
: : *
T0

O
cond_14/switch_tIdentitycond_14/Switch:1*
T0
*
_output_shapes
: 
M
cond_14/switch_fIdentitycond_14/Switch*
_output_shapes
: *
T0

H
cond_14/pred_idIdentity
Greater_13*
T0
*
_output_shapes
: 

cond_14/truediv/Cast/SwitchSwitchcount_nonzero_18/Sumcond_14/pred_id*'
_class
loc:@count_nonzero_18/Sum*
_output_shapes
: : *
T0	
k
cond_14/truediv/CastCastcond_14/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	

cond_14/truediv/Cast_1/SwitchSwitchcount_nonzero_20/Sumcond_14/pred_id*
T0	*'
_class
loc:@count_nonzero_20/Sum*
_output_shapes
: : 
o
cond_14/truediv/Cast_1Castcond_14/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
i
cond_14/truedivRealDivcond_14/truediv/Castcond_14/truediv/Cast_1*
_output_shapes
: *
T0
i
cond_14/ConstConst^cond_14/switch_f*
valueB 2        *
_output_shapes
: *
dtype0
b
cond_14/MergeMergecond_14/Constcond_14/truediv*
_output_shapes
: : *
T0*
N
V
recall_6/tagsConst*
valueB Brecall_6*
_output_shapes
: *
dtype0
X
recall_6ScalarSummaryrecall_6/tagscond_14/Merge*
T0*
_output_shapes
: 
Z
total_loss/tagsConst*
dtype0*
_output_shapes
: *
valueB B
total_loss
W

total_lossScalarSummarytotal_loss/tagsloss/Sum*
_output_shapes
: *
T0
J
lr/tagsConst*
value
B Blr*
dtype0*
_output_shapes
: 
L
lrScalarSummarylr/tagslearning_rate*
_output_shapes
: *
T0
Z
accuracy_1/tagsConst*
valueB B
accuracy_1*
_output_shapes
: *
dtype0
`

accuracy_1ScalarSummaryaccuracy_1/tagsaccuracy/accuracy*
T0*
_output_shapes
: 

Merge/MergeSummaryMergeSummaryprecision_0recall_0precision_1recall_1precision_2recall_2precision_3recall_3precision_4recall_4precision_5recall_5precision_6recall_6
total_losslr
accuracy_1*
_output_shapes
: *
N
^
total_loss_1/tagsConst*
valueB Btotal_loss_1*
dtype0*
_output_shapes
: 
[
total_loss_1ScalarSummarytotal_loss_1/tagsloss/Sum*
_output_shapes
: *
T0
N
	lr_1/tagsConst*
valueB
 Blr_1*
_output_shapes
: *
dtype0
P
lr_1ScalarSummary	lr_1/tagslearning_rate*
T0*
_output_shapes
: 
Y
Merge_1/MergeSummaryMergeSummarytotal_loss_1lr_1*
_output_shapes
: *
N
P

save/ConstConst*
valueB Bmodel*
_output_shapes
: *
dtype0
Ń

save/SaveV2/tensor_namesConst*

valueś	B÷	$BVariableBbeta1_powerBbeta2_powerBencoder/initial_state_0Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1Binput_embedBinput_embed/AdamBinput_embed/Adam_1Boutput_projection/WBoutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/bBoutput_projection/b/AdamBoutput_projection/b/Adam_1*
dtype0*
_output_shapes
:$
«
save/SaveV2/shape_and_slicesConst*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:$
ü

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariablebeta1_powerbeta2_powerencoder/initial_state_0encoder/initial_state_0/Adamencoder/initial_state_0/Adam_1encoder/initial_state_1encoder/initial_state_1/Adamencoder/initial_state_1/Adam_1encoder/initial_state_2encoder/initial_state_2/Adamencoder/initial_state_2/Adam_1encoder/initial_state_3encoder/initial_state_3/Adamencoder/initial_state_3/Adam_12encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_13encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_12encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_13encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1input_embedinput_embed/Adaminput_embed/Adam_1output_projection/Woutput_projection/W/Adamoutput_projection/W/Adam_1output_projection/boutput_projection/b/Adamoutput_projection/b/Adam_1*2
dtypes(
&2$
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_output_shapes
: *
_class
loc:@save/Const
l
save/RestoreV2/tensor_namesConst*
valueBBVariable*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:

save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
q
save/RestoreV2_1/tensor_namesConst*
dtype0*
_output_shapes
:* 
valueBBbeta1_power
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
 
save/Assign_1Assignbeta1_powersave/RestoreV2_1*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@input_embed
q
save/RestoreV2_2/tensor_namesConst*
_output_shapes
:*
dtype0* 
valueBBbeta2_power
j
!save/RestoreV2_2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
 
save/Assign_2Assignbeta2_powersave/RestoreV2_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@input_embed
}
save/RestoreV2_3/tensor_namesConst*
dtype0*
_output_shapes
:*,
value#B!Bencoder/initial_state_0
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
Į
save/Assign_3Assignencoder/initial_state_0save/RestoreV2_3**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

save/RestoreV2_4/tensor_namesConst*
dtype0*
_output_shapes
:*1
value(B&Bencoder/initial_state_0/Adam
j
!save/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
Ę
save/Assign_4Assignencoder/initial_state_0/Adamsave/RestoreV2_4*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0

save/RestoreV2_5/tensor_namesConst*
_output_shapes
:*
dtype0*3
value*B(Bencoder/initial_state_0/Adam_1
j
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
Č
save/Assign_5Assignencoder/initial_state_0/Adam_1save/RestoreV2_5*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0
}
save/RestoreV2_6/tensor_namesConst*
dtype0*
_output_shapes
:*,
value#B!Bencoder/initial_state_1
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Į
save/Assign_6Assignencoder/initial_state_1save/RestoreV2_6*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1

save/RestoreV2_7/tensor_namesConst*
dtype0*
_output_shapes
:*1
value(B&Bencoder/initial_state_1/Adam
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
Ę
save/Assign_7Assignencoder/initial_state_1/Adamsave/RestoreV2_7*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1

save/RestoreV2_8/tensor_namesConst*
_output_shapes
:*
dtype0*3
value*B(Bencoder/initial_state_1/Adam_1
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
Č
save/Assign_8Assignencoder/initial_state_1/Adam_1save/RestoreV2_8*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	
}
save/RestoreV2_9/tensor_namesConst*
_output_shapes
:*
dtype0*,
value#B!Bencoder/initial_state_2
j
!save/RestoreV2_9/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
Į
save/Assign_9Assignencoder/initial_state_2save/RestoreV2_9*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2

save/RestoreV2_10/tensor_namesConst*
dtype0*
_output_shapes
:*1
value(B&Bencoder/initial_state_2/Adam
k
"save/RestoreV2_10/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
Č
save/Assign_10Assignencoder/initial_state_2/Adamsave/RestoreV2_10*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_2*
T0*
use_locking(

save/RestoreV2_11/tensor_namesConst*3
value*B(Bencoder/initial_state_2/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save/Assign_11Assignencoder/initial_state_2/Adam_1save/RestoreV2_11*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_2*
validate_shape(*
_output_shapes
:	
~
save/RestoreV2_12/tensor_namesConst*,
value#B!Bencoder/initial_state_3*
dtype0*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ć
save/Assign_12Assignencoder/initial_state_3save/RestoreV2_12*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_3*
T0*
use_locking(

save/RestoreV2_13/tensor_namesConst*
dtype0*
_output_shapes
:*1
value(B&Bencoder/initial_state_3/Adam
k
"save/RestoreV2_13/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
Č
save/Assign_13Assignencoder/initial_state_3/Adamsave/RestoreV2_13**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

save/RestoreV2_14/tensor_namesConst*3
value*B(Bencoder/initial_state_3/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
_output_shapes
:*
dtypes
2
Ź
save/Assign_14Assignencoder/initial_state_3/Adam_1save/RestoreV2_14*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3

save/RestoreV2_15/tensor_namesConst*
_output_shapes
:*
dtype0*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
k
"save/RestoreV2_15/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
_output_shapes
:*
dtypes
2
õ
save/Assign_15Assign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasessave/RestoreV2_15*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(

save/RestoreV2_16/tensor_namesConst*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
ś
save/Assign_16Assign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adamsave/RestoreV2_16*
_output_shapes	
:*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
T0*
use_locking(
 
save/RestoreV2_17/tensor_namesConst*
dtype0*
_output_shapes
:*N
valueEBCB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1
k
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
ü
save/Assign_17Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1save/RestoreV2_17*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
validate_shape(*
_output_shapes	
:

save/RestoreV2_18/tensor_namesConst*H
value?B=B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
_output_shapes
:*
dtype0
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
ü
save/Assign_18Assign3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightssave/RestoreV2_18* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0*
use_locking(

save/RestoreV2_19/tensor_namesConst*
dtype0*
_output_shapes
:*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:

save/Assign_19Assign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamsave/RestoreV2_19*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
”
save/RestoreV2_20/tensor_namesConst*
_output_shapes
:*
dtype0*O
valueFBDB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1
k
"save/RestoreV2_20/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_20Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1save/RestoreV2_20* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0*
use_locking(

save/RestoreV2_21/tensor_namesConst*
dtype0*
_output_shapes
:*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
k
"save/RestoreV2_21/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
_output_shapes
:*
dtypes
2
õ
save/Assign_21Assign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasessave/RestoreV2_21*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
validate_shape(*
_output_shapes	
:

save/RestoreV2_22/tensor_namesConst*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_22/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save/Assign_22Assign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adamsave/RestoreV2_22*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
 
save/RestoreV2_23/tensor_namesConst*
_output_shapes
:*
dtype0*N
valueEBCB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1
k
"save/RestoreV2_23/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
ü
save/Assign_23Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1save/RestoreV2_23*
_output_shapes	
:*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
use_locking(

save/RestoreV2_24/tensor_namesConst*H
value?B=B3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
_output_shapes
:
k
"save/RestoreV2_24/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
_output_shapes
:*
dtypes
2
ü
save/Assign_24Assign3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightssave/RestoreV2_24* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0*
use_locking(

save/RestoreV2_25/tensor_namesConst*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_25/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_25Assign8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adamsave/RestoreV2_25*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
”
save/RestoreV2_26/tensor_namesConst*O
valueFBDB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:

save/Assign_26Assign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1save/RestoreV2_26*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
r
save/RestoreV2_27/tensor_namesConst*
dtype0*
_output_shapes
:* 
valueBBinput_embed
k
"save/RestoreV2_27/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
Æ
save/Assign_27Assigninput_embedsave/RestoreV2_27*#
_output_shapes
:*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
w
save/RestoreV2_28/tensor_namesConst*
_output_shapes
:*
dtype0*%
valueBBinput_embed/Adam
k
"save/RestoreV2_28/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
“
save/Assign_28Assigninput_embed/Adamsave/RestoreV2_28*#
_output_shapes
:*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
y
save/RestoreV2_29/tensor_namesConst*'
valueBBinput_embed/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
_output_shapes
:*
dtypes
2
¶
save/Assign_29Assigninput_embed/Adam_1save/RestoreV2_29*#
_output_shapes
:*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
z
save/RestoreV2_30/tensor_namesConst*(
valueBBoutput_projection/W*
_output_shapes
:*
dtype0
k
"save/RestoreV2_30/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
_output_shapes
:*
dtypes
2
»
save/Assign_30Assignoutput_projection/Wsave/RestoreV2_30*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

save/RestoreV2_31/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Boutput_projection/W/Adam
k
"save/RestoreV2_31/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
_output_shapes
:*
dtypes
2
Ą
save/Assign_31Assignoutput_projection/W/Adamsave/RestoreV2_31*
use_locking(*
T0*&
_class
loc:@output_projection/W*
validate_shape(*
_output_shapes
:	

save/RestoreV2_32/tensor_namesConst*/
value&B$Boutput_projection/W/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_32/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
Ā
save/Assign_32Assignoutput_projection/W/Adam_1save/RestoreV2_32*
use_locking(*
T0*&
_class
loc:@output_projection/W*
validate_shape(*
_output_shapes
:	
z
save/RestoreV2_33/tensor_namesConst*
dtype0*
_output_shapes
:*(
valueBBoutput_projection/b
k
"save/RestoreV2_33/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
_output_shapes
:*
dtypes
2
¶
save/Assign_33Assignoutput_projection/bsave/RestoreV2_33*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*&
_class
loc:@output_projection/b

save/RestoreV2_34/tensor_namesConst*
_output_shapes
:*
dtype0*-
value$B"Boutput_projection/b/Adam
k
"save/RestoreV2_34/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
»
save/Assign_34Assignoutput_projection/b/Adamsave/RestoreV2_34*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*&
_class
loc:@output_projection/b

save/RestoreV2_35/tensor_namesConst*
dtype0*
_output_shapes
:*/
value&B$Boutput_projection/b/Adam_1
k
"save/RestoreV2_35/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
½
save/Assign_35Assignoutput_projection/b/Adam_1save/RestoreV2_35*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*&
_class
loc:@output_projection/b
š
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35

4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedVariable*
_output_shapes
: *
dtype0*
_class
loc:@Variable
”
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedinput_embed*
_class
loc:@input_embed*
_output_shapes
: *
dtype0
¹
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedencoder/initial_state_0*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_0
¹
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedencoder/initial_state_1*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_1
¹
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedencoder/initial_state_2*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_2
¹
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedencoder/initial_state_3*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_3
ń
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
_output_shapes
: *
dtype0
ļ
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes
: *
dtype0
ń
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
_output_shapes
: *
dtype0
ļ
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitialized2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
²
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedoutput_projection/W*&
_class
loc:@output_projection/W*
dtype0*
_output_shapes
: 
²
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedoutput_projection/b*
_output_shapes
: *
dtype0*&
_class
loc:@output_projection/b
¢
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedbeta1_power*
_class
loc:@input_embed*
dtype0*
_output_shapes
: 
¢
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedbeta2_power*
_class
loc:@input_embed*
_output_shapes
: *
dtype0
§
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedinput_embed/Adam*
_class
loc:@input_embed*
dtype0*
_output_shapes
: 
©
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedinput_embed/Adam_1*
dtype0*
_output_shapes
: *
_class
loc:@input_embed
æ
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedencoder/initial_state_0/Adam*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_0
Į
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitializedencoder/initial_state_0/Adam_1**
_class 
loc:@encoder/initial_state_0*
_output_shapes
: *
dtype0
æ
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitializedencoder/initial_state_1/Adam*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_1
Į
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitializedencoder/initial_state_1/Adam_1*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_1
æ
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitializedencoder/initial_state_2/Adam**
_class 
loc:@encoder/initial_state_2*
_output_shapes
: *
dtype0
Į
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitializedencoder/initial_state_2/Adam_1*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_2
æ
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitializedencoder/initial_state_3/Adam**
_class 
loc:@encoder/initial_state_3*
dtype0*
_output_shapes
: 
Į
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitializedencoder/initial_state_3/Adam_1*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_3
÷
7report_uninitialized_variables/IsVariableInitialized_24IsVariableInitialized8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*
_output_shapes
: *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
ł
7report_uninitialized_variables/IsVariableInitialized_25IsVariableInitialized:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
õ
7report_uninitialized_variables/IsVariableInitialized_26IsVariableInitialized7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
dtype0*
_output_shapes
: 
÷
7report_uninitialized_variables/IsVariableInitialized_27IsVariableInitialized9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
dtype0*
_output_shapes
: 
÷
7report_uninitialized_variables/IsVariableInitialized_28IsVariableInitialized8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
_output_shapes
: 
ł
7report_uninitialized_variables/IsVariableInitialized_29IsVariableInitialized:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
_output_shapes
: 
õ
7report_uninitialized_variables/IsVariableInitialized_30IsVariableInitialized7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*
dtype0*
_output_shapes
: *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
÷
7report_uninitialized_variables/IsVariableInitialized_31IsVariableInitialized9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
·
7report_uninitialized_variables/IsVariableInitialized_32IsVariableInitializedoutput_projection/W/Adam*&
_class
loc:@output_projection/W*
dtype0*
_output_shapes
: 
¹
7report_uninitialized_variables/IsVariableInitialized_33IsVariableInitializedoutput_projection/W/Adam_1*
_output_shapes
: *
dtype0*&
_class
loc:@output_projection/W
·
7report_uninitialized_variables/IsVariableInitialized_34IsVariableInitializedoutput_projection/b/Adam*
_output_shapes
: *
dtype0*&
_class
loc:@output_projection/b
¹
7report_uninitialized_variables/IsVariableInitialized_35IsVariableInitializedoutput_projection/b/Adam_1*&
_class
loc:@output_projection/b*
dtype0*
_output_shapes
: 
Ž
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_217report_uninitialized_variables/IsVariableInitialized_227report_uninitialized_variables/IsVariableInitialized_237report_uninitialized_variables/IsVariableInitialized_247report_uninitialized_variables/IsVariableInitialized_257report_uninitialized_variables/IsVariableInitialized_267report_uninitialized_variables/IsVariableInitialized_277report_uninitialized_variables/IsVariableInitialized_287report_uninitialized_variables/IsVariableInitialized_297report_uninitialized_variables/IsVariableInitialized_307report_uninitialized_variables/IsVariableInitialized_317report_uninitialized_variables/IsVariableInitialized_327report_uninitialized_variables/IsVariableInitialized_337report_uninitialized_variables/IsVariableInitialized_347report_uninitialized_variables/IsVariableInitialized_35*
N$*
T0
*
_output_shapes
:$*

axis 
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:$
Ż

$report_uninitialized_variables/ConstConst*

valueś	B÷	$BVariableBinput_embedBencoder/initial_state_0Bencoder/initial_state_1Bencoder/initial_state_2Bencoder/initial_state_3B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesBoutput_projection/WBoutput_projection/bBbeta1_powerBbeta2_powerBinput_embed/AdamBinput_embed/Adam_1Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1B7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1Boutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/b/AdamBoutput_projection/b/Adam_1*
dtype0*
_output_shapes
:$
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
dtype0*
_output_shapes
:*
valueB:$

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ł
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
end_mask *

begin_mask*
ellipsis_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
Index0*
T0

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
õ
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
valueB:$*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
į
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
shrink_axis_mask *
_output_shapes
: *
T0*
Index0*
end_mask*
new_axis_mask *
ellipsis_mask *

begin_mask 
Æ
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
N*
T0*
_output_shapes
:*

axis 
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
«
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Ė
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
T0*
_output_shapes
:$*
Tshape0

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:
Ū
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
T0
*
_output_shapes
:$*
Tshape0

1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:’’’’’’’’’
¶
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:’’’’’’’’’*
T0	

2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:’’’’’’’’’

initNoOp^Variable/Assign^input_embed/Assign^encoder/initial_state_0/Assign^encoder/initial_state_1/Assign^encoder/initial_state_2/Assign^encoder/initial_state_3/Assign;^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Assign:^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Assign;^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Assign:^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Assign^output_projection/W/Assign^output_projection/b/Assign^beta1_power/Assign^beta2_power/Assign^input_embed/Adam/Assign^input_embed/Adam_1/Assign$^encoder/initial_state_0/Adam/Assign&^encoder/initial_state_0/Adam_1/Assign$^encoder/initial_state_1/Adam/Assign&^encoder/initial_state_1/Adam_1/Assign$^encoder/initial_state_2/Adam/Assign&^encoder/initial_state_2/Adam_1/Assign$^encoder/initial_state_3/Adam/Assign&^encoder/initial_state_3/Adam_1/Assign@^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/AssignB^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/Assign?^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/AssignA^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/Assign@^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/AssignB^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/Assign?^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/AssignA^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/Assign ^output_projection/W/Adam/Assign"^output_projection/W/Adam_1/Assign ^output_projection/b/Adam/Assign"^output_projection/b/Adam_1/Assign

init_1NoOp

init_all_tablesNoOp
-

group_depsNoOp^init_1^init_all_tables"D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"
init_op

init"O
cond_context’NüN

batch_and_pad/cond/cond_textbatch_and_pad/cond/pred_id:0batch_and_pad/cond/switch_t:0 *°
'batch_and_pad/cond/control_dependency:0
6batch_and_pad/cond/padding_fifo_queue_enqueue/Switch:1
8batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_1:1
8batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_2:1
8batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_3:1
batch_and_pad/cond/pred_id:0
batch_and_pad/cond/switch_t:0
"batch_and_pad/padding_fifo_queue:0
random_queue_test_Dequeue:0
random_queue_test_Dequeue:1
random_queue_test_Dequeue:2\
"batch_and_pad/padding_fifo_queue:06batch_and_pad/cond/padding_fifo_queue_enqueue/Switch:1W
random_queue_test_Dequeue:28batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_3:1W
random_queue_test_Dequeue:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_2:1W
random_queue_test_Dequeue:08batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_1:1
Ē
batch_and_pad/cond/cond_text_1batch_and_pad/cond/pred_id:0batch_and_pad/cond/switch_f:0*h
)batch_and_pad/cond/control_dependency_1:0
batch_and_pad/cond/pred_id:0
batch_and_pad/cond/switch_f:0
¶
batch_and_pad_1/cond/cond_textbatch_and_pad_1/cond/pred_id:0batch_and_pad_1/cond/switch_t:0 *Š
)batch_and_pad_1/cond/control_dependency:0
8batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch:1
:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_1:1
:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_2:1
:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_3:1
batch_and_pad_1/cond/pred_id:0
batch_and_pad_1/cond/switch_t:0
$batch_and_pad_1/padding_fifo_queue:0
random_queue_train_Dequeue:0
random_queue_train_Dequeue:1
random_queue_train_Dequeue:2Z
random_queue_train_Dequeue:0:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_1:1Z
random_queue_train_Dequeue:1:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_2:1Z
random_queue_train_Dequeue:2:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_3:1`
$batch_and_pad_1/padding_fifo_queue:08batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch:1
Ó
 batch_and_pad_1/cond/cond_text_1batch_and_pad_1/cond/pred_id:0batch_and_pad_1/cond/switch_f:0*n
+batch_and_pad_1/cond/control_dependency_1:0
batch_and_pad_1/cond/pred_id:0
batch_and_pad_1/cond/switch_f:0
č
cond/cond_textcond/pred_id:0cond/switch_t:0 *²
batch_and_pad_1:0
batch_and_pad_1:1
batch_and_pad_1:2
cond/Switch_1:0
cond/Switch_1:1
cond/Switch_2:0
cond/Switch_2:1
cond/Switch_3:0
cond/Switch_3:1
cond/pred_id:0
cond/switch_t:0$
batch_and_pad_1:0cond/Switch_1:1$
batch_and_pad_1:2cond/Switch_3:1$
batch_and_pad_1:1cond/Switch_2:1
Ü
cond/cond_text_1cond/pred_id:0cond/switch_f:0*¦
batch_and_pad:0
batch_and_pad:1
batch_and_pad:2
cond/Switch_4:0
cond/Switch_4:1
cond/Switch_5:0
cond/Switch_5:1
cond/Switch_6:0
cond/Switch_6:1
cond/pred_id:0
cond/switch_f:0"
batch_and_pad:2cond/Switch_6:0"
batch_and_pad:0cond/Switch_4:0"
batch_and_pad:1cond/Switch_5:0
ū
cond_1/cond_textcond_1/pred_id:0cond_1/switch_t:0 *æ
cond_1/pred_id:0
cond_1/switch_t:0
cond_1/truediv/Cast/Switch:1
cond_1/truediv/Cast:0
cond_1/truediv/Cast_1/Switch:1
cond_1/truediv/Cast_1:0
cond_1/truediv:0
count_nonzero/Sum:0
count_nonzero_1/Sum:03
count_nonzero/Sum:0cond_1/truediv/Cast/Switch:17
count_nonzero_1/Sum:0cond_1/truediv/Cast_1/Switch:1
p
cond_1/cond_text_1cond_1/pred_id:0cond_1/switch_f:0*5
cond_1/Const:0
cond_1/pred_id:0
cond_1/switch_f:0
ū
cond_2/cond_textcond_2/pred_id:0cond_2/switch_t:0 *æ
cond_2/pred_id:0
cond_2/switch_t:0
cond_2/truediv/Cast/Switch:1
cond_2/truediv/Cast:0
cond_2/truediv/Cast_1/Switch:1
cond_2/truediv/Cast_1:0
cond_2/truediv:0
count_nonzero/Sum:0
count_nonzero_2/Sum:03
count_nonzero/Sum:0cond_2/truediv/Cast/Switch:17
count_nonzero_2/Sum:0cond_2/truediv/Cast_1/Switch:1
p
cond_2/cond_text_1cond_2/pred_id:0cond_2/switch_f:0*5
cond_2/Const:0
cond_2/pred_id:0
cond_2/switch_f:0
’
cond_3/cond_textcond_3/pred_id:0cond_3/switch_t:0 *Ć
cond_3/pred_id:0
cond_3/switch_t:0
cond_3/truediv/Cast/Switch:1
cond_3/truediv/Cast:0
cond_3/truediv/Cast_1/Switch:1
cond_3/truediv/Cast_1:0
cond_3/truediv:0
count_nonzero_3/Sum:0
count_nonzero_4/Sum:05
count_nonzero_3/Sum:0cond_3/truediv/Cast/Switch:17
count_nonzero_4/Sum:0cond_3/truediv/Cast_1/Switch:1
p
cond_3/cond_text_1cond_3/pred_id:0cond_3/switch_f:0*5
cond_3/Const:0
cond_3/pred_id:0
cond_3/switch_f:0
’
cond_4/cond_textcond_4/pred_id:0cond_4/switch_t:0 *Ć
cond_4/pred_id:0
cond_4/switch_t:0
cond_4/truediv/Cast/Switch:1
cond_4/truediv/Cast:0
cond_4/truediv/Cast_1/Switch:1
cond_4/truediv/Cast_1:0
cond_4/truediv:0
count_nonzero_3/Sum:0
count_nonzero_5/Sum:05
count_nonzero_3/Sum:0cond_4/truediv/Cast/Switch:17
count_nonzero_5/Sum:0cond_4/truediv/Cast_1/Switch:1
p
cond_4/cond_text_1cond_4/pred_id:0cond_4/switch_f:0*5
cond_4/Const:0
cond_4/pred_id:0
cond_4/switch_f:0
’
cond_5/cond_textcond_5/pred_id:0cond_5/switch_t:0 *Ć
cond_5/pred_id:0
cond_5/switch_t:0
cond_5/truediv/Cast/Switch:1
cond_5/truediv/Cast:0
cond_5/truediv/Cast_1/Switch:1
cond_5/truediv/Cast_1:0
cond_5/truediv:0
count_nonzero_6/Sum:0
count_nonzero_7/Sum:07
count_nonzero_7/Sum:0cond_5/truediv/Cast_1/Switch:15
count_nonzero_6/Sum:0cond_5/truediv/Cast/Switch:1
p
cond_5/cond_text_1cond_5/pred_id:0cond_5/switch_f:0*5
cond_5/Const:0
cond_5/pred_id:0
cond_5/switch_f:0
’
cond_6/cond_textcond_6/pred_id:0cond_6/switch_t:0 *Ć
cond_6/pred_id:0
cond_6/switch_t:0
cond_6/truediv/Cast/Switch:1
cond_6/truediv/Cast:0
cond_6/truediv/Cast_1/Switch:1
cond_6/truediv/Cast_1:0
cond_6/truediv:0
count_nonzero_6/Sum:0
count_nonzero_8/Sum:07
count_nonzero_8/Sum:0cond_6/truediv/Cast_1/Switch:15
count_nonzero_6/Sum:0cond_6/truediv/Cast/Switch:1
p
cond_6/cond_text_1cond_6/pred_id:0cond_6/switch_f:0*5
cond_6/Const:0
cond_6/pred_id:0
cond_6/switch_f:0

cond_7/cond_textcond_7/pred_id:0cond_7/switch_t:0 *Å
cond_7/pred_id:0
cond_7/switch_t:0
cond_7/truediv/Cast/Switch:1
cond_7/truediv/Cast:0
cond_7/truediv/Cast_1/Switch:1
cond_7/truediv/Cast_1:0
cond_7/truediv:0
count_nonzero_10/Sum:0
count_nonzero_9/Sum:08
count_nonzero_10/Sum:0cond_7/truediv/Cast_1/Switch:15
count_nonzero_9/Sum:0cond_7/truediv/Cast/Switch:1
p
cond_7/cond_text_1cond_7/pred_id:0cond_7/switch_f:0*5
cond_7/Const:0
cond_7/pred_id:0
cond_7/switch_f:0

cond_8/cond_textcond_8/pred_id:0cond_8/switch_t:0 *Å
cond_8/pred_id:0
cond_8/switch_t:0
cond_8/truediv/Cast/Switch:1
cond_8/truediv/Cast:0
cond_8/truediv/Cast_1/Switch:1
cond_8/truediv/Cast_1:0
cond_8/truediv:0
count_nonzero_11/Sum:0
count_nonzero_9/Sum:08
count_nonzero_11/Sum:0cond_8/truediv/Cast_1/Switch:15
count_nonzero_9/Sum:0cond_8/truediv/Cast/Switch:1
p
cond_8/cond_text_1cond_8/pred_id:0cond_8/switch_f:0*5
cond_8/Const:0
cond_8/pred_id:0
cond_8/switch_f:0

cond_9/cond_textcond_9/pred_id:0cond_9/switch_t:0 *Ē
cond_9/pred_id:0
cond_9/switch_t:0
cond_9/truediv/Cast/Switch:1
cond_9/truediv/Cast:0
cond_9/truediv/Cast_1/Switch:1
cond_9/truediv/Cast_1:0
cond_9/truediv:0
count_nonzero_12/Sum:0
count_nonzero_13/Sum:08
count_nonzero_13/Sum:0cond_9/truediv/Cast_1/Switch:16
count_nonzero_12/Sum:0cond_9/truediv/Cast/Switch:1
p
cond_9/cond_text_1cond_9/pred_id:0cond_9/switch_f:0*5
cond_9/Const:0
cond_9/pred_id:0
cond_9/switch_f:0

cond_10/cond_textcond_10/pred_id:0cond_10/switch_t:0 *Š
cond_10/pred_id:0
cond_10/switch_t:0
cond_10/truediv/Cast/Switch:1
cond_10/truediv/Cast:0
cond_10/truediv/Cast_1/Switch:1
cond_10/truediv/Cast_1:0
cond_10/truediv:0
count_nonzero_12/Sum:0
count_nonzero_14/Sum:09
count_nonzero_14/Sum:0cond_10/truediv/Cast_1/Switch:17
count_nonzero_12/Sum:0cond_10/truediv/Cast/Switch:1
v
cond_10/cond_text_1cond_10/pred_id:0cond_10/switch_f:0*8
cond_10/Const:0
cond_10/pred_id:0
cond_10/switch_f:0

cond_11/cond_textcond_11/pred_id:0cond_11/switch_t:0 *Š
cond_11/pred_id:0
cond_11/switch_t:0
cond_11/truediv/Cast/Switch:1
cond_11/truediv/Cast:0
cond_11/truediv/Cast_1/Switch:1
cond_11/truediv/Cast_1:0
cond_11/truediv:0
count_nonzero_15/Sum:0
count_nonzero_16/Sum:09
count_nonzero_16/Sum:0cond_11/truediv/Cast_1/Switch:17
count_nonzero_15/Sum:0cond_11/truediv/Cast/Switch:1
v
cond_11/cond_text_1cond_11/pred_id:0cond_11/switch_f:0*8
cond_11/Const:0
cond_11/pred_id:0
cond_11/switch_f:0

cond_12/cond_textcond_12/pred_id:0cond_12/switch_t:0 *Š
cond_12/pred_id:0
cond_12/switch_t:0
cond_12/truediv/Cast/Switch:1
cond_12/truediv/Cast:0
cond_12/truediv/Cast_1/Switch:1
cond_12/truediv/Cast_1:0
cond_12/truediv:0
count_nonzero_15/Sum:0
count_nonzero_17/Sum:07
count_nonzero_15/Sum:0cond_12/truediv/Cast/Switch:19
count_nonzero_17/Sum:0cond_12/truediv/Cast_1/Switch:1
v
cond_12/cond_text_1cond_12/pred_id:0cond_12/switch_f:0*8
cond_12/Const:0
cond_12/pred_id:0
cond_12/switch_f:0

cond_13/cond_textcond_13/pred_id:0cond_13/switch_t:0 *Š
cond_13/pred_id:0
cond_13/switch_t:0
cond_13/truediv/Cast/Switch:1
cond_13/truediv/Cast:0
cond_13/truediv/Cast_1/Switch:1
cond_13/truediv/Cast_1:0
cond_13/truediv:0
count_nonzero_18/Sum:0
count_nonzero_19/Sum:07
count_nonzero_18/Sum:0cond_13/truediv/Cast/Switch:19
count_nonzero_19/Sum:0cond_13/truediv/Cast_1/Switch:1
v
cond_13/cond_text_1cond_13/pred_id:0cond_13/switch_f:0*8
cond_13/Const:0
cond_13/pred_id:0
cond_13/switch_f:0

cond_14/cond_textcond_14/pred_id:0cond_14/switch_t:0 *Š
cond_14/pred_id:0
cond_14/switch_t:0
cond_14/truediv/Cast/Switch:1
cond_14/truediv/Cast:0
cond_14/truediv/Cast_1/Switch:1
cond_14/truediv/Cast_1:0
cond_14/truediv:0
count_nonzero_18/Sum:0
count_nonzero_20/Sum:09
count_nonzero_20/Sum:0cond_14/truediv/Cast_1/Switch:17
count_nonzero_18/Sum:0cond_14/truediv/Cast/Switch:1
v
cond_14/cond_text_1cond_14/pred_id:0cond_14/switch_f:0*8
cond_14/Const:0
cond_14/pred_id:0
cond_14/switch_f:0"
train_op

Adam"D
ready_op8
6
4report_uninitialized_variables/boolean_mask/Gather:0"ń
	summariesć
ą
%batch_and_pad/fraction_of_1384_full:0
'batch_and_pad_1/fraction_of_1384_full:0
grad_norms/grad_norms:0
precision_0:0

recall_0:0
precision_1:0

recall_1:0
precision_2:0

recall_2:0
precision_3:0

recall_3:0
precision_4:0

recall_4:0
precision_5:0

recall_5:0
precision_6:0

recall_6:0
total_loss:0
lr:0
accuracy_1:0
total_loss_1:0
lr_1:0"¬

trainable_variables


7
input_embed:0input_embed/Assigninput_embed/read:0
[
encoder/initial_state_0:0encoder/initial_state_0/Assignencoder/initial_state_0/read:0
[
encoder/initial_state_1:0encoder/initial_state_1/Assignencoder/initial_state_1/read:0
[
encoder/initial_state_2:0encoder/initial_state_2/Assignencoder/initial_state_2/read:0
[
encoder/initial_state_3:0encoder/initial_state_3/Assignencoder/initial_state_3/read:0
Æ
5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights:0:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read:0
¬
4encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases:09encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read:0
Æ
5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights:0:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Assign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read:0
¬
4encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases:09encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/read:0
O
output_projection/W:0output_projection/W/Assignoutput_projection/W/read:0
O
output_projection/b:0output_projection/b/Assignoutput_projection/b/read:0"É
queue_runners·“

 batch_and_pad/padding_fifo_queuebatch_and_pad/cond/Merge:0&batch_and_pad/padding_fifo_queue_Close"(batch_and_pad/padding_fifo_queue_Close_1*

"batch_and_pad_1/padding_fifo_queuebatch_and_pad_1/cond/Merge:0(batch_and_pad_1/padding_fifo_queue_Close"*batch_and_pad_1/padding_fifo_queue_Close_1*"
local_init_op


group_deps"
while_context
’
(encoder_1/rnn/while/encoder_1/rnn/while/ *encoder_1/rnn/while/LoopCond:02encoder_1/rnn/while/Merge:0:encoder_1/rnn/while/Identity:0Bencoder_1/rnn/while/Exit:0Bencoder_1/rnn/while/Exit_1:0Bencoder_1/rnn/while/Exit_2:0Bencoder_1/rnn/while/Exit_3:0Bencoder_1/rnn/while/Exit_4:0Bencoder_1/rnn/while/Exit_5:0Bgradients/f_count_2:0J§
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read:0
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read:0
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/read:0
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read:0
encoder_1/rnn/CheckSeqLen:0
encoder_1/rnn/TensorArray:0
Jencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
encoder_1/rnn/TensorArray_1:0
encoder_1/rnn/strided_slice_2:0
encoder_1/rnn/while/Enter:0
encoder_1/rnn/while/Enter_1:0
encoder_1/rnn/while/Enter_2:0
encoder_1/rnn/while/Enter_3:0
encoder_1/rnn/while/Enter_4:0
encoder_1/rnn/while/Enter_5:0
encoder_1/rnn/while/Exit:0
encoder_1/rnn/while/Exit_1:0
encoder_1/rnn/while/Exit_2:0
encoder_1/rnn/while/Exit_3:0
encoder_1/rnn/while/Exit_4:0
encoder_1/rnn/while/Exit_5:0
(encoder_1/rnn/while/GreaterEqual/Enter:0
"encoder_1/rnn/while/GreaterEqual:0
$encoder_1/rnn/while/GreaterEqual_1:0
$encoder_1/rnn/while/GreaterEqual_2:0
$encoder_1/rnn/while/GreaterEqual_3:0
$encoder_1/rnn/while/GreaterEqual_4:0
encoder_1/rnn/while/Identity:0
 encoder_1/rnn/while/Identity_1:0
 encoder_1/rnn/while/Identity_2:0
 encoder_1/rnn/while/Identity_3:0
 encoder_1/rnn/while/Identity_4:0
 encoder_1/rnn/while/Identity_5:0
 encoder_1/rnn/while/Less/Enter:0
encoder_1/rnn/while/Less:0
encoder_1/rnn/while/LoopCond:0
encoder_1/rnn/while/Merge:0
encoder_1/rnn/while/Merge:1
encoder_1/rnn/while/Merge_1:0
encoder_1/rnn/while/Merge_1:1
encoder_1/rnn/while/Merge_2:0
encoder_1/rnn/while/Merge_2:1
encoder_1/rnn/while/Merge_3:0
encoder_1/rnn/while/Merge_3:1
encoder_1/rnn/while/Merge_4:0
encoder_1/rnn/while/Merge_4:1
encoder_1/rnn/while/Merge_5:0
encoder_1/rnn/while/Merge_5:1
#encoder_1/rnn/while/NextIteration:0
%encoder_1/rnn/while/NextIteration_1:0
%encoder_1/rnn/while/NextIteration_2:0
%encoder_1/rnn/while/NextIteration_3:0
%encoder_1/rnn/while/NextIteration_4:0
%encoder_1/rnn/while/NextIteration_5:0
"encoder_1/rnn/while/Select/Enter:0
encoder_1/rnn/while/Select:0
encoder_1/rnn/while/Select_1:0
encoder_1/rnn/while/Select_2:0
encoder_1/rnn/while/Select_3:0
encoder_1/rnn/while/Select_4:0
encoder_1/rnn/while/Switch:0
encoder_1/rnn/while/Switch:1
encoder_1/rnn/while/Switch_1:0
encoder_1/rnn/while/Switch_1:1
encoder_1/rnn/while/Switch_2:0
encoder_1/rnn/while/Switch_2:1
encoder_1/rnn/while/Switch_3:0
encoder_1/rnn/while/Switch_3:1
encoder_1/rnn/while/Switch_4:0
encoder_1/rnn/while/Switch_4:1
encoder_1/rnn/while/Switch_5:0
encoder_1/rnn/while/Switch_5:1
-encoder_1/rnn/while/TensorArrayReadV3/Enter:0
/encoder_1/rnn/while/TensorArrayReadV3/Enter_1:0
'encoder_1/rnn/while/TensorArrayReadV3:0
?encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
9encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
encoder_1/rnn/while/add/y:0
encoder_1/rnn/while/add:0
Cencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter:0
=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd:0
=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid:0
?encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1:0
?encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2:0
:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh:0
<encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1:0
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add/y:0
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add:0
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1:0
Lencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter:0
Fencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul:0
Kencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis:0
Fencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat:0
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul:0
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1:0
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2:0
Eencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim:0
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:0
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:1
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:2
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:3
Cencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter:0
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd:0
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid:0
?encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1:0
?encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2:0
:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh:0
<encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1:0
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add/y:0
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add:0
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1:0
Lencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter:0
Fencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul:0
Kencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis:0
Fencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat:0
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul:0
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1:0
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2:0
Eencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim:0
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:0
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:1
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:2
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:3
encoder_1/rnn/zeros:0
gradients/Add/y:0
gradients/Add:0
gradients/Merge:0
gradients/Merge:1
gradients/NextIteration:0
gradients/Switch:0
gradients/Switch:1
=gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnter:0
>gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPush:0
:gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc:0
Agradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnter:0
Bgradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPush:0
>gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc:0
=gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnter:0
>gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPush:0
:gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc:0
Agradients/encoder_1/rnn/while/Select_2_grad/zeros_like/RefEnter:0
Bgradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPush:0
>gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc:0
=gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnter:0
>gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPush:0
:gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc:0
Agradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnter:0
Bgradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPush:0
>gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc:0
=gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnter:0
>gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPush:0
:gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc:0
Agradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnter:0
Bgradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPush:0
>gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc:0
cgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnter:0
dgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPush:0
`gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc:0
ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnter:0
hgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush:0
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter:0
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPush:0
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc:0
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Shape_1:0
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnter:0
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPush:0
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc:0
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnter:0
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPush:0
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc:0
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnter:0
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPush:0
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc:0
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnter:0
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPush:0
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc:0
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnter:0
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPush:0
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc:0
ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnter:0
hgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush:0
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1:0
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnter:0
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPush:0
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc:0
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnter:0
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPush:0
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc:0
ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnter:0
hgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush:0
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter:0
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPush:0
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc:0
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Shape_1:0
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnter:0
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPush:0
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc:0
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnter:0
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPush:0
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc:0
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnter:0
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPush:0
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc:0
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnter:0
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPush:0
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc:0
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnter:0
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPush:0
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc:0
ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnter:0
hgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush:0
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1:0
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnter:0
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPush:0
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc:0
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnter:0
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPush:0
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc:0
gradients/f_count:0
gradients/f_count_1:0
gradients/f_count_2:0
>gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc:0Agradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnter:0Ļ
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnter:0
>gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc:0Agradients/encoder_1/rnn/while/Select_2_grad/zeros_like/RefEnter:0{
:gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc:0=gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnter:0Ļ
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc:0ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnter:0
>gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc:0Agradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnter:0
>gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc:0Agradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnter:0µ
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc:0Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnter:0³
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc:0Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnter:0³
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc:0Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnter:0G
encoder_1/rnn/CheckSeqLen:0(encoder_1/rnn/while/GreaterEqual/Enter:0Æ
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnter:0Æ
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnter:0Æ
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnter:0{
:gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc:0=gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnter:0C
encoder_1/rnn/strided_slice_2:0 encoder_1/rnn/while/Less/Enter:0Ļ
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnter:0Ļ
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc:0ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnter:0
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read:0Cencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter:0Ē
`gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc:0cgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnter:0
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/read:0Cencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter:0Ė
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc:0egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter:0³
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc:0Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnter:0{
:gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc:0=gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnter:0;
encoder_1/rnn/zeros:0"encoder_1/rnn/while/Select/Enter:0³
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc:0Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnter:0
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read:0Lencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter:0Æ
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnter:0Æ
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnter:0N
encoder_1/rnn/TensorArray_1:0-encoder_1/rnn/while/TensorArrayReadV3/Enter:0Å
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc:0bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnter:0
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read:0Lencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter:0}
Jencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0/encoder_1/rnn/while/TensorArrayReadV3/Enter_1:0^
encoder_1/rnn/TensorArray:0?encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0{
:gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc:0=gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnter:0µ
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc:0Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnter:0Ė
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc:0egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter:0Æ
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnter:0Å
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc:0bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnter:0"ņ"
	variablesä"į"
.

Variable:0Variable/AssignVariable/read:0
7
input_embed:0input_embed/Assigninput_embed/read:0
[
encoder/initial_state_0:0encoder/initial_state_0/Assignencoder/initial_state_0/read:0
[
encoder/initial_state_1:0encoder/initial_state_1/Assignencoder/initial_state_1/read:0
[
encoder/initial_state_2:0encoder/initial_state_2/Assignencoder/initial_state_2/read:0
[
encoder/initial_state_3:0encoder/initial_state_3/Assignencoder/initial_state_3/read:0
Æ
5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights:0:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read:0
¬
4encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases:09encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read:0
Æ
5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights:0:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Assign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read:0
¬
4encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases:09encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/read:0
O
output_projection/W:0output_projection/W/Assignoutput_projection/W/read:0
O
output_projection/b:0output_projection/b/Assignoutput_projection/b/read:0
7
beta1_power:0beta1_power/Assignbeta1_power/read:0
7
beta2_power:0beta2_power/Assignbeta2_power/read:0
F
input_embed/Adam:0input_embed/Adam/Assigninput_embed/Adam/read:0
L
input_embed/Adam_1:0input_embed/Adam_1/Assigninput_embed/Adam_1/read:0
j
encoder/initial_state_0/Adam:0#encoder/initial_state_0/Adam/Assign#encoder/initial_state_0/Adam/read:0
p
 encoder/initial_state_0/Adam_1:0%encoder/initial_state_0/Adam_1/Assign%encoder/initial_state_0/Adam_1/read:0
j
encoder/initial_state_1/Adam:0#encoder/initial_state_1/Adam/Assign#encoder/initial_state_1/Adam/read:0
p
 encoder/initial_state_1/Adam_1:0%encoder/initial_state_1/Adam_1/Assign%encoder/initial_state_1/Adam_1/read:0
j
encoder/initial_state_2/Adam:0#encoder/initial_state_2/Adam/Assign#encoder/initial_state_2/Adam/read:0
p
 encoder/initial_state_2/Adam_1:0%encoder/initial_state_2/Adam_1/Assign%encoder/initial_state_2/Adam_1/read:0
j
encoder/initial_state_3/Adam:0#encoder/initial_state_3/Adam/Assign#encoder/initial_state_3/Adam/read:0
p
 encoder/initial_state_3/Adam_1:0%encoder/initial_state_3/Adam_1/Assign%encoder/initial_state_3/Adam_1/read:0
¾
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam:0?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/Assign?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/read:0
Ä
<encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1:0Aencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/AssignAencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/read:0
»
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam:0>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/Assign>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/read:0
Į
;encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1:0@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/Assign@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/read:0
¾
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam:0?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/Assign?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/read:0
Ä
<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1:0Aencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/AssignAencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/read:0
»
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam:0>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/Assign>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/read:0
Į
;encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1:0@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/Assign@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/read:0
^
output_projection/W/Adam:0output_projection/W/Adam/Assignoutput_projection/W/Adam/read:0
d
output_projection/W/Adam_1:0!output_projection/W/Adam_1/Assign!output_projection/W/Adam_1/read:0
^
output_projection/b/Adam:0output_projection/b/Adam/Assignoutput_projection/b/Adam/read:0
d
output_projection/b/Adam_1:0!output_projection/b/Adam_1/Assign!output_projection/b/Adam_1/read:0ŠÖi       <7ø4	¹CÖA:;ga.        )ķ©P	ļCÖA*

Variable/sec    ĪxĶ6       OWļ	ū®”CÖA:+'logs/s2l_2017-05-06_13-29-45/model.ckptĶ.G²#       °wC	$NCÖA®*

Variable/sec(w·?R'Ä9       7ń	LŽ£NCÖAÆ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt2£O#       °wC	„CÖAÜ*

Variable/secw·?ŌŅ%9       7ń	\¾CÖAŻ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt«FU#       °wC	Ó-äCÖA
*

Variable/sec³Qø?Ņn?9       7ń	¼y¤äCÖA
:+'logs/s2l_2017-05-06_13-29-45/model.ckpt±P#       °wC	ėp/CÖAŗ*

Variable/sec×v·?hīĀ9       7ń	-©/CÖA»:+'logs/s2l_2017-05-06_13-29-45/model.ckptxgŚ#       °wC	`zCÖAź*

Variable/secģRø?ß’ź9       7ń	£„zCÖAź:+'logs/s2l_2017-05-06_13-29-45/model.ckptŅ2&#       °wC	ĀÅCÖA*

Variable/secæø?[ü9       7ń	ØÅCÖA:+'logs/s2l_2017-05-06_13-29-45/model.ckpt5#       °wC	ĘżCÖAĖ*

Variable/sec!Rø?JÓa9       7ń	Ē*®CÖAĢ:+'logs/s2l_2017-05-06_13-29-45/model.ckptN.#       °wC	[CÖAū*

Variable/secōMø?]Å@9       7ń	é¦[CÖAü:+'logs/s2l_2017-05-06_13-29-45/model.ckpt#·ŗ)#       °wC	ī&¦CÖA«*

Variable/sec~Uø?Ķń9       7ń	tŃŖ¦CÖA«:+'logs/s2l_2017-05-06_13-29-45/model.ckpt}^_#       °wC	.9ńCÖA×!*

Variable/secŲ¶?ŃĻ1ń9       7ń	čØńCÖAŲ!:+'logs/s2l_2017-05-06_13-29-45/model.ckpt/#       °wC	ü’<CÖAū$*

Variable/secŗ3³?ļn9       7ń	Ć¤<CÖAü$:+'logs/s2l_2017-05-06_13-29-45/model.ckptRÅxī#       °wC	­üCÖA„(*

Variable/secĀµ?öŁ¼9       7ń	,¦CÖA„(:+'logs/s2l_2017-05-06_13-29-45/model.ckpt ²#       °wC	Ü*ŅCÖAÖ+*

Variable/sec²¾ø?5ß.R9       7ń	N*¬ŅCÖAÖ+:+'logs/s2l_2017-05-06_13-29-45/model.ckptõx«9#       °wC	JCÖA/*

Variable/sec
·?\Æ9       7ń	ē¤CÖA/:+'logs/s2l_2017-05-06_13-29-45/model.ckptŚö’##       °wC	hCÖA³2*

Variable/sec»Qø?1Ī39       7ń	Q”hCÖA“2:+'logs/s2l_2017-05-06_13-29-45/model.ckpt·Ēųč#       °wC	ž³CÖAß5*

Variable/secC¶?Ė0R 9       7ń	)Ø³CÖAą5:+'logs/s2l_2017-05-06_13-29-45/model.ckpt÷1)#       °wC	²*žCÖA9*

Variable/sec¹¾ø?c6įŪ9       7ń	GćØžCÖA9:+'logs/s2l_2017-05-06_13-29-45/model.ckpte1D#       °wC	yICÖAń;*

Variable/secQ?o:9       7ń	)æICÖAņ;:+'logs/s2l_2017-05-06_13-29-45/model.ckptéŪ#       °wC	}CÖAĶ>*

Variable/secßz?Ćł9       7ń	Q”æCÖAĪ>:+'logs/s2l_2017-05-06_13-29-45/model.ckptPXÉ#       °wC	HūßCÖAč@*

Variable/seco~q?
¼ż9       7ń	ō1­ßCÖAé@:+'logs/s2l_2017-05-06_13-29-45/model.ckpt¾GĪ#       °wC	ś*CÖAC*

Variable/sec|ww?×¼ņ9       7ń	G“*CÖAC:+'logs/s2l_2017-05-06_13-29-45/model.ckpt“|ńŹ#       °wC	ĪüuCÖA¦E*

Variable/secµXr?ž9       7ń	ČuCÖA§E:+'logs/s2l_2017-05-06_13-29-45/model.ckptęS=Ā#       °wC	/ĄCÖAĀG*

Variable/sec Xr?šo§9       7ń	Ū²ĄCÖAĀG:+'logs/s2l_2017-05-06_13-29-45/model.ckptÓcT#       °wC	CÖAÜI*

Variable/sec4¤p?Ćuh9       7ń	D±CÖAŻI:+'logs/s2l_2017-05-06_13-29-45/model.ckpt6Ķ#       °wC	 VCÖAL*

Variable/secz?oūOq9       7ń	åG°VCÖAL:+'logs/s2l_2017-05-06_13-29-45/model.ckptų}¼’#       °wC	¼”CÖA·N*

Variable/sec8D?KG¦9       7ń	F·”CÖA·N:+'logs/s2l_2017-05-06_13-29-45/model.ckpt-%(#       °wC	[ģCÖAļP*

Variable/sec#?ņjģ9       7ń	ōQµģCÖAļP:+'logs/s2l_2017-05-06_13-29-45/model.ckptūöö#       °wC	ļ7CÖAØS*

Variable/sec?ó¤å9       7ń	n÷Ą7CÖA©S:+'logs/s2l_2017-05-06_13-29-45/model.ckptY?æ~#       °wC	ICÖAįU*

Variable/secÖ?±Ü½k9       7ń	h©CÖAįU:+'logs/s2l_2017-05-06_13-29-45/model.ckptT.3ź#       °wC	dĶCÖAX*

Variable/secŠ? o³Ó9       7ń	ĘĶCÖAX:+'logs/s2l_2017-05-06_13-29-45/model.ckpt4ć#       °wC	 CÖAŌZ*

Variable/sec6f?° \Č9       7ń	§tŖCÖAŌZ:+'logs/s2l_2017-05-06_13-29-45/model.ckptBĆq#       °wC	&cCÖA]*

Variable/secÆ?$9       7ń	Ė.©cCÖA]:+'logs/s2l_2017-05-06_13-29-45/model.ckptē#       °wC	¤®CÖAĆ_*

Variable/sec±?Ļ“æ
9       7ń	h[¬®CÖAÄ_:+'logs/s2l_2017-05-06_13-29-45/model.ckpt
Će#       °wC	łCÖAža*

Variable/sectf?iŅų9       7ń	ś¬łCÖAža:+'logs/s2l_2017-05-06_13-29-45/model.ckptŽFL%#       °wC	·$DCÖA¶d*

Variable/sec?`#·L9       7ń	æ³DCÖA¶d:+'logs/s2l_2017-05-06_13-29-45/model.ckptnø;#       °wC	%CÖAäf*

Variable/sec¦Ś?8abÉ9       7ń	ßŖCÖAäf:+'logs/s2l_2017-05-06_13-29-45/model.ckptS~z#       °wC	āŚCÖAi*

Variable/secņ“?ŌNŲH9       7ń	9|ŃŚCÖAi:+'logs/s2l_2017-05-06_13-29-45/model.ckpt*²#       °wC	¬%CÖAĒk*

Variable/secgü?ś&ĘR9       7ń	Z¬%CÖAČk:+'logs/s2l_2017-05-06_13-29-45/model.ckpt“Ń#       °wC	É"pCÖAn*

Variable/sec]f?9»59       7ń	ńØpCÖAn:+'logs/s2l_2017-05-06_13-29-45/model.ckptĖIxf#       °wC	Ó-»CÖA¼p*

Variable/secł?ĮW 9       7ń	l&Æ»CÖA¼p:+'logs/s2l_2017-05-06_13-29-45/model.ckpt,Yæ#       °wC	­CÖA÷r*

Variable/sece?ņl¢9       7ń	©CÖA÷r:+'logs/s2l_2017-05-06_13-29-45/model.ckpt<#»#       °wC	ŅQCÖAÆu*

Variable/sec¹?)G9       7ń	"³QCÖAÆu:+'logs/s2l_2017-05-06_13-29-45/model.ckptĆĄč#       °wC		CÖAŅw*

Variable/sec+Rx?w[Ę/9       7ń	7»CÖAÓw:+'logs/s2l_2017-05-06_13-29-45/model.ckptźĻ#       °wC	ēCÖAüy*

Variable/secŽJ~?9Øį.9       7ń	÷ ÄēCÖAży:+'logs/s2l_2017-05-06_13-29-45/model.ckpt®.²#       °wC	2CÖA|*

Variable/sec¦ww?w$9       7ń	,¬2CÖA|:+'logs/s2l_2017-05-06_13-29-45/model.ckpt{FÕ`#       °wC	ā}CÖAÕ~*

Variable/sec±?Ń²ÄŁ9       7ń	3N»}CÖAÖ~:+'logs/s2l_2017-05-06_13-29-45/model.ckpt«z>$       B+M	OČCÖA*

Variable/secAm?Ģa¼„:       źā	EöŖČCÖA:+'logs/s2l_2017-05-06_13-29-45/model.ckptķĒ$       B+M	VCÖA±*

Variable/secG?ZÕja:       źā	Öü¶CÖA²:+'logs/s2l_2017-05-06_13-29-45/model.ckptĶ8k$       B+M	ś^CÖAē*

Variable/secMD?Āpŗ:       źā	GÆ¹^CÖAč:+'logs/s2l_2017-05-06_13-29-45/model.ckpt`"ę@$       B+M	©CÖA*

Variable/sec¦ü?ģ»:       źā	2 Ć©CÖA:+'logs/s2l_2017-05-06_13-29-45/model.ckptš$       B+M	²ōCÖAŌ*

Variable/secł?2š±:       źā	³°ōCÖAÕ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt!e'²$       B+M	+?CÖA*

Variable/secv±?ļ9ū:       źā	ĆF±?CÖA:+'logs/s2l_2017-05-06_13-29-45/model.ckptĪŁP$       B+M	wCÖAĒ*

Variable/sec¬Ó?ŁOEŗ:       źā	ų4«CÖAĒ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt5¾óÅ$       B+M	£ÕCÖA*

Variable/sec?ł?ė8:       źā	ń»ÕCÖA:+'logs/s2l_2017-05-06_13-29-45/model.ckptŌš$       B+M	ŗ CÖAŗ*

Variable/secŽ?ÖvŚÜ:       źā	šMŖ CÖA»:+'logs/s2l_2017-05-06_13-29-45/model.ckptvŌ$       B+M	zkCÖAō*

Variable/sec#ł?Ŗ}":       źā	žĢkCÖAõ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt ĮB$       B+M	"¶CÖA¬*

Variable/sec¤?ā`ü:       źā	eŗ¶CÖA­:+'logs/s2l_2017-05-06_13-29-45/model.ckptTsÆ$       B+M	&CÖAą*

Variable/secČi? Å:       źā	ōĆCÖAį:+'logs/s2l_2017-05-06_13-29-45/model.ckptĒ4r$       B+M	ŚLCÖA*

Variable/seca"?Ż|×:       źā	°LCÖA:+'logs/s2l_2017-05-06_13-29-45/model.ckptćiA5$       B+M	-$CÖAĄ *

Variable/sec{G?Wėģ:       źā	:±²CÖAĄ :+'logs/s2l_2017-05-06_13-29-45/model.ckptņ_nR$       B+M	0āCÖA÷¢*

Variable/sec±? DćZ:       źā	j„ĮāCÖAų¢:+'logs/s2l_2017-05-06_13-29-45/model.ckpt³o$       B+M	&-CÖA­„*

Variable/secD?cĢ[:       źā	]¬-CÖA­„:+'logs/s2l_2017-05-06_13-29-45/model.ckptH4*F$       B+M	VxCÖAį§*

Variable/secj?UMC:       źā	åøxCÖAā§:+'logs/s2l_2017-05-06_13-29-45/model.ckptnn_$       B+M	CĆCÖAŖ*

Variable/secŁi?Éõ:       źā	£±¬ĆCÖAŖ:+'logs/s2l_2017-05-06_13-29-45/model.ckptõā-$       B+M	“CÖAŹ¬*

Variable/sec×?uź>:       źā	«CÖAŹ¬:+'logs/s2l_2017-05-06_13-29-45/model.ckpt^ļš$       B+M	1'YCÖA’®*

Variable/secŹÖ?1!:       źā	9`ĢYCÖAÆ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt8£!_$       B+M	Õ¤CÖA±*

Variable/secčt?_Q:       źā	.©®¤CÖA±:+'logs/s2l_2017-05-06_13-29-45/model.ckptwóĄ$       B+M	OļCÖA¾³*

Variable/secĀu?ø@.:       źā	r§ÆļCÖA¾³:+'logs/s2l_2017-05-06_13-29-45/model.ckptå$       B+M	Ę:CÖAćµ*

Variable/secz?ßŁ:       źā	Ļ.®:CÖAäµ:+'logs/s2l_2017-05-06_13-29-45/model.ckptg|$       B+M	ž
CÖAø*

Variable/secŹ?ĒŹ×ģ:       źā	ŖCÖAø:+'logs/s2l_2017-05-06_13-29-45/model.ckptyą)Ž$       B+M	ŠCÖAŹŗ*

Variable/secG?äk :       źā	}æŠCÖAĖŗ:+'logs/s2l_2017-05-06_13-29-45/model.ckptv$       B+M	-CÖAų¼*

Variable/sec”Ś?*SĶ:       źā	nśØCÖAų¼:+'logs/s2l_2017-05-06_13-29-45/model.ckpts?t$       B+M	9fCÖA«æ*

Variable/secü?öe:       źā	ßÄ­fCÖA«æ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt--ņ4$       B+M	Ļ±CÖAßĮ*

Variable/secĀi?Ūķ:       źā	ėÉ«±CÖAßĮ:+'logs/s2l_2017-05-06_13-29-45/model.ckptĪ	_$       B+M	¹üCÖAÄ*

Variable/secbf?R=:       źā	1©üCÖAÄ:+'logs/s2l_2017-05-06_13-29-45/model.ckptqs$       B+M	:GCÖAÓĘ*

Variable/sec?Pä{\:       źā	ŖGCÖAÓĘ:+'logs/s2l_2017-05-06_13-29-45/model.ckptä°ä$       B+M	\CÖAÉ*

Variable/secGD?ļ·C:       źā	A¹CÖAÉ:+'logs/s2l_2017-05-06_13-29-45/model.ckptōč$       B+M	YŻCÖAÄĖ*

Variable/secĶe?eŌžē:       źā	ŗ²ŻCÖAÄĖ:+'logs/s2l_2017-05-06_13-29-45/model.ckptįß$       B+M	\(CÖAžĶ*

Variable/secĆł?¤Fw„:       źā	aś³(CÖA’Ķ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt±iz$       B+M	UūsCÖAµŠ*

Variable/sec±?Ļ¹J:       źā	ļ©sCÖAµŠ:+'logs/s2l_2017-05-06_13-29-45/model.ckptoQē"$       B+M	¾CÖAķŅ*

Variable/sec?Å«:       źā	ń*Ŗ¾CÖAķŅ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt¦xŠØ$       B+M	O	CÖA§Õ*

Variable/secVł?Ā«7W:       źā	K«	CÖA§Õ:+'logs/s2l_2017-05-06_13-29-45/model.ckptµ¢¶$       B+M	žTCÖAÜ×*

Variable/secłÖ??u7):       źā	*«TCÖAŻ×:+'logs/s2l_2017-05-06_13-29-45/model.ckpt"ŅĖ$       B+M	ĒCÖAŚ*

Variable/secpŚ?³|:       źā	äiŲCÖAŚ:+'logs/s2l_2017-05-06_13-29-45/model.ckpté6Ŗ$       B+M	¤ źCÖA¾Ü*

Variable/secéi?"Ēø
:       źā	Ä[¬źCÖA¾Ü:+'logs/s2l_2017-05-06_13-29-45/model.ckpt¾1JŁ$       B+M	n5CÖAņŽ*

Variable/sec³i?·RL:       źā	~¬Æ5CÖAóŽ:+'logs/s2l_2017-05-06_13-29-45/model.ckpth0é$       B+M	ŲCÖA®į*

Variable/secØÓ?³E1:       źā	Ń ¹CÖAÆį:+'logs/s2l_2017-05-06_13-29-45/model.ckptõHSÕ$       B+M	'ĖCÖA¤ä*

Variable/sec?_šy\:       źā	a¦ĖCÖA¤ä:+'logs/s2l_2017-05-06_13-29-45/model.ckpt	¦$       B+M	M.CÖAÕē*

Variable/secæø?Dimŗ:       źā	6ZÆCÖAÕē:+'logs/s2l_2017-05-06_13-29-45/model.ckptM¶ų$       B+M	üaCÖAė*

Variable/sec,¹?Ńtøµ:       źā	?9¦aCÖAė:+'logs/s2l_2017-05-06_13-29-45/model.ckpt1śč$       B+M	F¬CÖAī*

Variable/secóĢ¬?JF;ø:       źā	aß„¬CÖAī:+'logs/s2l_2017-05-06_13-29-45/model.ckptz¤$       B+M	ē÷CÖA÷š*

Variable/sec?÷#:       źā	ø®÷CÖA÷š:+'logs/s2l_2017-05-06_13-29-45/model.ckptęé($       B+M	ÜBCÖA„ó*

Variable/secŚ?!ég:       źā	xÆBCÖA„ó:+'logs/s2l_2017-05-06_13-29-45/model.ckptwe$       B+M	ĪCÖAįõ*

Variable/secÓ?ė÷):       źā	Ŗ+¼CÖAįõ:+'logs/s2l_2017-05-06_13-29-45/model.ckptW$       B+M	²bŲCÖAų*

Variable/secp“?(ą©Ļ:       źā	ta»ŲCÖAų:+'logs/s2l_2017-05-06_13-29-45/model.ckptä%!č$       B+M	Ō
#CÖAĀś*

Variable/sec»"?HŲ×:       źā	ż¢°#CÖAĀś:+'logs/s2l_2017-05-06_13-29-45/model.ckpt<ō$       B+M	nCÖAöü*

Variable/secÓi?C~µ:       źā	ÆnCÖAöü:+'logs/s2l_2017-05-06_13-29-45/model.ckpt½ł$       B+M	\¹CÖAŖ’*

Variable/sec­i?ŁŅ“,:       źā	ßkĀ¹CÖA«’:+'logs/s2l_2017-05-06_13-29-45/model.ckptĒo$       B+M	CÖAé*

Variable/secn?×łĘ:       źā	.ŹæCÖAé:+'logs/s2l_2017-05-06_13-29-45/model.ckpt¦'ß$       B+M	pOCÖA„*

Variable/sec§Ó?_Ó¹:       źā	]j¬OCÖA¦:+'logs/s2l_2017-05-06_13-29-45/model.ckptāS)$       B+M	uCÖAą*

Variable/secmf?ģx:       źā	ŠŃ°CÖAą:+'logs/s2l_2017-05-06_13-29-45/model.ckptäsv$       B+M	0åCÖA*

Variable/secł?t#/:       źā	²­åCÖA:+'logs/s2l_2017-05-06_13-29-45/model.ckpt:ŲYļ$       B+M	®0 CÖAŌ*

Variable/sec.ł?¾øNJ:       źā	X9¾0 CÖAÕ:+'logs/s2l_2017-05-06_13-29-45/model.ckptęŃ$       B+M	Y{ CÖA*

Variable/secß?MŹ:       źā	÷Ą{ CÖA:+'logs/s2l_2017-05-06_13-29-45/model.ckptPQ0®$       B+M	÷!Ę CÖAĒ*

Variable/secł?šÕd:       źā	D÷µĘ CÖAĒ:+'logs/s2l_2017-05-06_13-29-45/model.ckptgC:f$       B+M	6”CÖA*

Variable/secł?Vž¹Ų:       źā	VÖŖ”CÖA:+'logs/s2l_2017-05-06_13-29-45/model.ckptż½Ē$       B+M	\”CÖAØ*

Variable/sec¼{?A“5:       źā	kÓ\”CÖA©:+'logs/s2l_2017-05-06_13-29-45/model.ckptßĀX5$       B+M	 §”CÖAÖ*

Variable/sechŚ?iā:       źā	jĆ§”CÖA×:+'logs/s2l_2017-05-06_13-29-45/model.ckpt°ß$       B+M	Aņ”CÖA*

Variable/secjD?'ZO:       źā	¹ņ”CÖA:+'logs/s2l_2017-05-06_13-29-45/model.ckptōÓ$       B+M	=¢CÖAĀ*

Variable/secBD?@
už:       źā	ä ¹=¢CÖAĆ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt Sŗ]$       B+M	Lś¢CÖAž*

Variable/sec¾Ó?ą×:       źā	·Ą¢CÖA’:+'logs/s2l_2017-05-06_13-29-45/model.ckptļt=$       B+M	øÓ¢CÖA»”*

Variable/secÅ@?ām:       źā	K«ÉÓ¢CÖA¼”:+'logs/s2l_2017-05-06_13-29-45/model.ckpt¾öB$       B+M	±ł£CÖAõ£*

Variable/secEł?ą-:       źā	"ø£CÖAõ£:+'logs/s2l_2017-05-06_13-29-45/model.ckpt é6$       B+M	ri£CÖA­¦*

Variable/sec?²k©e:       źā	fĮi£CÖA®¦:+'logs/s2l_2017-05-06_13-29-45/model.ckpt	Ķ$       B+M	“£CÖA×Ø*

Variable/secK~?}Ź:       źā	©Śø“£CÖAŲØ:+'logs/s2l_2017-05-06_13-29-45/model.ckptĀ!ģ$       B+M	-’£CÖA«*

Variable/sec(×?yQæ:       źā	ÕxÆ’£CÖA«:+'logs/s2l_2017-05-06_13-29-45/model.ckpt”,$       B+M	ĪūJ¤CÖAÅ­*

Variable/sec?l_é:       źā	©æJ¤CÖAĘ­:+'logs/s2l_2017-05-06_13-29-45/model.ckpt¾åōq$       B+M	²¤CÖAņÆ*

Variable/secm?<ęų_:       źā	XĘ¬¤CÖAņÆ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt³Śm$       B+M	Ųbą¤CÖA”²*

Variable/secG?ÓIåZ:       źā	¾ųĻą¤CÖA¢²:+'logs/s2l_2017-05-06_13-29-45/model.ckptķhõ&$       B+M	e+„CÖA×“*

Variable/secĮD?lh,:       źā	Ė¾“+„CÖAŲ“:+'logs/s2l_2017-05-06_13-29-45/model.ckptBkó;$       B+M	v„CÖA·*

Variable/secż??į:       źā	ü„©v„CÖA·:+'logs/s2l_2017-05-06_13-29-45/model.ckptU²¤0$       B+M	āĮ„CÖAĖ¹*

Variable/secQ²?Č%r:       źā	µ«Į„CÖAĢ¹:+'logs/s2l_2017-05-06_13-29-45/model.ckpt^$qi$       B+M	h¦CÖA¼*

Variable/secf?Łb:       źā	©¦CÖA¼:+'logs/s2l_2017-05-06_13-29-45/model.ckptcN$       B+M	²gW¦CÖAĮ¾*

Variable/sec±e?Le