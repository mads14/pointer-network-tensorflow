       �K"	   L�C�Abrain.Event:2�!V>     �fg	��L�C�A"��
f
PlaceholderPlaceholder*0
_output_shapes
:������������������*
dtype0*
shape: 
[
Placeholder_1Placeholder*#
_output_shapes
:���������*
shape: *
dtype0
X

early_stopPlaceholder*
shape: *
dtype0*#
_output_shapes
:���������
�
random_queue_testRandomShuffleQueueV2*#
shapes
:	�::*
	container *
seed2�*
shared_name *

seed{*
_output_shapes
: *
component_types
2*
min_after_dequeue�*
capacity�

�
random_queue_test_enqueueQueueEnqueueV2random_queue_testPlaceholderPlaceholder_1
early_stop*
Tcomponents
2*

timeout_ms���������
�
random_queue_test_DequeueQueueDequeueV2random_queue_test*

timeout_ms���������*+
_output_shapes
:	�::*
component_types
2
U
batch_and_pad/ConstConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
 batch_and_pad/padding_fifo_queuePaddingFIFOQueueV2*#
shapes
:	�::*
capacity�
*
	container *
shared_name *
_output_shapes
: *
component_types
2
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

�
4batch_and_pad/cond/padding_fifo_queue_enqueue/SwitchSwitch batch_and_pad/padding_fifo_queuebatch_and_pad/cond/pred_id*
_output_shapes
: : *3
_class)
'%loc:@batch_and_pad/padding_fifo_queue*
T0
�
6batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_1Switchrandom_queue_test_Dequeuebatch_and_pad/cond/pred_id*
T0*,
_class"
 loc:@random_queue_test_Dequeue**
_output_shapes
:	�:	�
�
6batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_2Switchrandom_queue_test_Dequeue:1batch_and_pad/cond/pred_id*,
_class"
 loc:@random_queue_test_Dequeue* 
_output_shapes
::*
T0
�
6batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_3Switchrandom_queue_test_Dequeue:2batch_and_pad/cond/pred_id*
T0* 
_output_shapes
::*,
_class"
 loc:@random_queue_test_Dequeue
�
-batch_and_pad/cond/padding_fifo_queue_enqueueQueueEnqueueV26batch_and_pad/cond/padding_fifo_queue_enqueue/Switch:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_1:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_2:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_3:1*
Tcomponents
2*

timeout_ms���������
�
%batch_and_pad/cond/control_dependencyIdentitybatch_and_pad/cond/switch_t.^batch_and_pad/cond/padding_fifo_queue_enqueue*.
_class$
" loc:@batch_and_pad/cond/switch_t*
_output_shapes
: *
T0

=
batch_and_pad/cond/NoOpNoOp^batch_and_pad/cond/switch_f
�
'batch_and_pad/cond/control_dependency_1Identitybatch_and_pad/cond/switch_f^batch_and_pad/cond/NoOp*
_output_shapes
: *.
_class$
" loc:@batch_and_pad/cond/switch_f*
T0

�
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
batch_and_pad/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *i=:
b
batch_and_pad/mulMulbatch_and_pad/Castbatch_and_pad/mul/y*
_output_shapes
: *
T0
�
(batch_and_pad/fraction_of_1384_full/tagsConst*
dtype0*
_output_shapes
: *4
value+B) B#batch_and_pad/fraction_of_1384_full
�
#batch_and_pad/fraction_of_1384_fullScalarSummary(batch_and_pad/fraction_of_1384_full/tagsbatch_and_pad/mul*
_output_shapes
: *
T0
R
batch_and_pad/nConst*
dtype0*
_output_shapes
: *
value
B :�
�
batch_and_padQueueDequeueManyV2 batch_and_pad/padding_fifo_queuebatch_and_pad/n*:
_output_shapes(
&:��:	�:	�*
component_types
2*

timeout_ms���������
h
Placeholder_2Placeholder*
shape: *
dtype0*0
_output_shapes
:������������������
[
Placeholder_3Placeholder*
shape: *
dtype0*#
_output_shapes
:���������
Z
early_stop_1Placeholder*#
_output_shapes
:���������*
dtype0*
shape: 
�
random_queue_trainRandomShuffleQueueV2*#
shapes
:	�::*
capacity�
*
min_after_dequeue�*
_output_shapes
: *
component_types
2*

seed{*
shared_name *
seed2{*
	container 
�
random_queue_train_enqueueQueueEnqueueV2random_queue_trainPlaceholder_2Placeholder_3early_stop_1*
Tcomponents
2*

timeout_ms���������
�
random_queue_train_DequeueQueueDequeueV2random_queue_train*

timeout_ms���������*+
_output_shapes
:	�::*
component_types
2
W
batch_and_pad_1/ConstConst*
value	B
 Z*
_output_shapes
: *
dtype0

�
"batch_and_pad_1/padding_fifo_queuePaddingFIFOQueueV2*#
shapes
:	�::*
capacity�
*
_output_shapes
: *
component_types
2*
shared_name *
	container 
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

�
6batch_and_pad_1/cond/padding_fifo_queue_enqueue/SwitchSwitch"batch_and_pad_1/padding_fifo_queuebatch_and_pad_1/cond/pred_id*
T0*5
_class+
)'loc:@batch_and_pad_1/padding_fifo_queue*
_output_shapes
: : 
�
8batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_1Switchrandom_queue_train_Dequeuebatch_and_pad_1/cond/pred_id**
_output_shapes
:	�:	�*-
_class#
!loc:@random_queue_train_Dequeue*
T0
�
8batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_2Switchrandom_queue_train_Dequeue:1batch_and_pad_1/cond/pred_id*-
_class#
!loc:@random_queue_train_Dequeue* 
_output_shapes
::*
T0
�
8batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_3Switchrandom_queue_train_Dequeue:2batch_and_pad_1/cond/pred_id*-
_class#
!loc:@random_queue_train_Dequeue* 
_output_shapes
::*
T0
�
/batch_and_pad_1/cond/padding_fifo_queue_enqueueQueueEnqueueV28batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch:1:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_1:1:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_2:1:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_3:1*
Tcomponents
2*

timeout_ms���������
�
'batch_and_pad_1/cond/control_dependencyIdentitybatch_and_pad_1/cond/switch_t0^batch_and_pad_1/cond/padding_fifo_queue_enqueue*0
_class&
$"loc:@batch_and_pad_1/cond/switch_t*
_output_shapes
: *
T0

A
batch_and_pad_1/cond/NoOpNoOp^batch_and_pad_1/cond/switch_f
�
)batch_and_pad_1/cond/control_dependency_1Identitybatch_and_pad_1/cond/switch_f^batch_and_pad_1/cond/NoOp*
T0
*
_output_shapes
: *0
_class&
$"loc:@batch_and_pad_1/cond/switch_f
�
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
batch_and_pad_1/mul/yConst*
valueB
 *i=:*
dtype0*
_output_shapes
: 
h
batch_and_pad_1/mulMulbatch_and_pad_1/Castbatch_and_pad_1/mul/y*
T0*
_output_shapes
: 
�
*batch_and_pad_1/fraction_of_1384_full/tagsConst*
_output_shapes
: *
dtype0*6
value-B+ B%batch_and_pad_1/fraction_of_1384_full
�
%batch_and_pad_1/fraction_of_1384_fullScalarSummary*batch_and_pad_1/fraction_of_1384_full/tagsbatch_and_pad_1/mul*
_output_shapes
: *
T0
T
batch_and_pad_1/nConst*
value
B :�*
_output_shapes
: *
dtype0
�
batch_and_pad_1QueueDequeueManyV2"batch_and_pad_1/padding_fifo_queuebatch_and_pad_1/n*

timeout_ms���������*:
_output_shapes(
&:��:	�:	�*
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
cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
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
cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
cond/Switch_1Switchbatch_and_pad_1cond/pred_id*
T0*"
_class
loc:@batch_and_pad_1*4
_output_shapes"
 :��:��
�
cond/Switch_2Switchbatch_and_pad_1:1cond/pred_id*
T0*"
_class
loc:@batch_and_pad_1**
_output_shapes
:	�:	�
�
cond/Switch_3Switchbatch_and_pad_1:2cond/pred_id**
_output_shapes
:	�:	�*"
_class
loc:@batch_and_pad_1*
T0
�
cond/Switch_4Switchbatch_and_padcond/pred_id*4
_output_shapes"
 :��:��* 
_class
loc:@batch_and_pad*
T0
�
cond/Switch_5Switchbatch_and_pad:1cond/pred_id* 
_class
loc:@batch_and_pad**
_output_shapes
:	�:	�*
T0
�
cond/Switch_6Switchbatch_and_pad:2cond/pred_id*
T0* 
_class
loc:@batch_and_pad**
_output_shapes
:	�:	�
m

cond/MergeMergecond/Switch_4cond/Switch_1:1*&
_output_shapes
:��: *
N*
T0
j
cond/Merge_1Mergecond/Switch_5cond/Switch_2:1*!
_output_shapes
:	�: *
T0*
N
j
cond/Merge_2Mergecond/Switch_6cond/Switch_3:1*!
_output_shapes
:	�: *
N*
T0
j
cond/Merge_3Mergecond/Switch_6cond/Switch_3:1*!
_output_shapes
:	�: *
N*
T0
`
Reshape/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
c
ReshapeReshapecond/Merge_2Reshape/shape*
Tshape0*
_output_shapes	
:�*
T0
X
Variable/initial_valueConst*
dtype0*
_output_shapes
: *
value	B : 
l
Variable
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
�
Variable/AssignAssignVariableVariable/initial_value*
_output_shapes
: *
validate_shape(*
_class
loc:@Variable*
T0*
use_locking(
a
Variable/readIdentityVariable*
_output_shapes
: *
_class
loc:@Variable*
T0
�
,input_embed/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
_class
loc:@input_embed*!
valueB"      �   
�
*input_embed/Initializer/random_uniform/minConst*
_class
loc:@input_embed*
valueB
 *
ף�*
dtype0*
_output_shapes
: 
�
*input_embed/Initializer/random_uniform/maxConst*
_class
loc:@input_embed*
valueB
 *
ף=*
_output_shapes
: *
dtype0
�
4input_embed/Initializer/random_uniform/RandomUniformRandomUniform,input_embed/Initializer/random_uniform/shape*#
_output_shapes
:�*
_class
loc:@input_embed*
dtype0*

seed{*
T0*
seed2W
�
*input_embed/Initializer/random_uniform/subSub*input_embed/Initializer/random_uniform/max*input_embed/Initializer/random_uniform/min*
_output_shapes
: *
_class
loc:@input_embed*
T0
�
*input_embed/Initializer/random_uniform/mulMul4input_embed/Initializer/random_uniform/RandomUniform*input_embed/Initializer/random_uniform/sub*
T0*
_class
loc:@input_embed*#
_output_shapes
:�
�
&input_embed/Initializer/random_uniformAdd*input_embed/Initializer/random_uniform/mul*input_embed/Initializer/random_uniform/min*
_class
loc:@input_embed*#
_output_shapes
:�*
T0
�
input_embed
VariableV2*
_class
loc:@input_embed*#
_output_shapes
:�*
shape:�*
dtype0*
shared_name *
	container 
�
input_embed/AssignAssigninput_embed&input_embed/Initializer/random_uniform*
_class
loc:@input_embed*#
_output_shapes
:�*
T0*
validate_shape(*
use_locking(
w
input_embed/readIdentityinput_embed*
_class
loc:@input_embed*#
_output_shapes
:�*
T0
_
encoder/conv1d/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
�
encoder/conv1d/ExpandDims
ExpandDims
cond/Mergeencoder/conv1d/ExpandDims/dim*(
_output_shapes
:��*
T0*

Tdim0
a
encoder/conv1d/ExpandDims_1/dimConst*
value	B : *
_output_shapes
: *
dtype0
�
encoder/conv1d/ExpandDims_1
ExpandDimsinput_embed/readencoder/conv1d/ExpandDims_1/dim*'
_output_shapes
:�*
T0*

Tdim0
�
encoder/conv1d/Conv2DConv2Dencoder/conv1d/ExpandDimsencoder/conv1d/ExpandDims_1*
use_cudnn_on_gpu(*
T0*
paddingVALID*)
_output_shapes
:���*
strides
*
data_formatNHWC

encoder/conv1d/SqueezeSqueezeencoder/conv1d/Conv2D*%
_output_shapes
:���*
T0*
squeeze_dims

Z
ShapeConst*!
valueB"�   �      *
_output_shapes
:*
dtype0
]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask*
Index0*
T0*
end_mask *
_output_shapes
: *

begin_mask *
ellipsis_mask 
�
)encoder/initial_state_0/Initializer/ConstConst*
dtype0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_0*
valueB	�*    
�
encoder/initial_state_0
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	�*
shape:	�*
shared_name 
�
encoder/initial_state_0/AssignAssignencoder/initial_state_0)encoder/initial_state_0/Initializer/Const*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_0*
validate_shape(*
_output_shapes
:	�
�
encoder/initial_state_0/readIdentityencoder/initial_state_0*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_0
m
+encoder_1/initial_state_0_tiled/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
)encoder_1/initial_state_0_tiled/multiplesPackstrided_slice+encoder_1/initial_state_0_tiled/multiples/1*
N*
T0*
_output_shapes
:*

axis 
�
encoder_1/initial_state_0_tiledTileencoder/initial_state_0/read)encoder_1/initial_state_0_tiled/multiples*

Tmultiples0*
T0*(
_output_shapes
:����������
�
)encoder/initial_state_1/Initializer/ConstConst**
_class 
loc:@encoder/initial_state_1*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
encoder/initial_state_1
VariableV2*
	container *
shared_name *
dtype0*
shape:	�*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_1
�
encoder/initial_state_1/AssignAssignencoder/initial_state_1)encoder/initial_state_1/Initializer/Const**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
encoder/initial_state_1/readIdentityencoder/initial_state_1**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	�*
T0
m
+encoder_1/initial_state_1_tiled/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
)encoder_1/initial_state_1_tiled/multiplesPackstrided_slice+encoder_1/initial_state_1_tiled/multiples/1*
_output_shapes
:*
N*

axis *
T0
�
encoder_1/initial_state_1_tiledTileencoder/initial_state_1/read)encoder_1/initial_state_1_tiled/multiples*

Tmultiples0*
T0*(
_output_shapes
:����������
�
)encoder/initial_state_2/Initializer/ConstConst*
dtype0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_2*
valueB	�*    
�
encoder/initial_state_2
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	�*
shape:	�*
shared_name 
�
encoder/initial_state_2/AssignAssignencoder/initial_state_2)encoder/initial_state_2/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_2
�
encoder/initial_state_2/readIdentityencoder/initial_state_2*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_2
m
+encoder_1/initial_state_2_tiled/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
)encoder_1/initial_state_2_tiled/multiplesPackstrided_slice+encoder_1/initial_state_2_tiled/multiples/1*
T0*

axis *
N*
_output_shapes
:
�
encoder_1/initial_state_2_tiledTileencoder/initial_state_2/read)encoder_1/initial_state_2_tiled/multiples*(
_output_shapes
:����������*
T0*

Tmultiples0
�
)encoder/initial_state_3/Initializer/ConstConst*
_output_shapes
:	�*
dtype0**
_class 
loc:@encoder/initial_state_3*
valueB	�*    
�
encoder/initial_state_3
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	�*
shape:	�*
shared_name 
�
encoder/initial_state_3/AssignAssignencoder/initial_state_3)encoder/initial_state_3/Initializer/Const*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	�
�
encoder/initial_state_3/readIdentityencoder/initial_state_3*
T0**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	�
m
+encoder_1/initial_state_3_tiled/multiples/1Const*
value	B :*
_output_shapes
: *
dtype0
�
)encoder_1/initial_state_3_tiled/multiplesPackstrided_slice+encoder_1/initial_state_3_tiled/multiples/1*
_output_shapes
:*
N*

axis *
T0
�
encoder_1/initial_state_3_tiledTileencoder/initial_state_3/read)encoder_1/initial_state_3_tiled/multiples*

Tmultiples0*
T0*(
_output_shapes
:����������
m
encoder_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
�
encoder_1/transpose	Transposeencoder/conv1d/Squeezeencoder_1/transpose/perm*
Tperm0*%
_output_shapes
:���*
T0
T
encoder_1/sequence_lengthIdentityReshape*
T0*
_output_shapes	
:�
h
encoder_1/rnn/ShapeConst*!
valueB"�   �   �   *
dtype0*
_output_shapes
:
k
!encoder_1/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
m
#encoder_1/rnn/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
m
#encoder_1/rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
encoder_1/rnn/strided_sliceStridedSliceencoder_1/rnn/Shape!encoder_1/rnn/strided_slice/stack#encoder_1/rnn/strided_slice/stack_1#encoder_1/rnn/strided_slice/stack_2*
Index0*
T0*
new_axis_mask *
_output_shapes
: *
shrink_axis_mask*

begin_mask *
ellipsis_mask *
end_mask 
m
#encoder_1/rnn/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
o
%encoder_1/rnn/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
o
%encoder_1/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
encoder_1/rnn/strided_slice_1StridedSliceencoder_1/rnn/Shape#encoder_1/rnn/strided_slice_1/stack%encoder_1/rnn/strided_slice_1/stack_1%encoder_1/rnn/strided_slice_1/stack_2*
_output_shapes
: *
end_mask *
new_axis_mask *
ellipsis_mask *

begin_mask *
shrink_axis_mask*
T0*
Index0
`
encoder_1/rnn/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�
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
encoder_1/rnn/AllAllencoder_1/rnn/Equalencoder_1/rnn/Const*

Tidx0*
	keep_dims( *
_output_shapes
: 
�
encoder_1/rnn/Assert/ConstConst*J
valueAB? B9Expected shape for Tensor encoder_1/sequence_length:0 is *
_output_shapes
: *
dtype0
m
encoder_1/rnn/Assert/Const_1Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
�
"encoder_1/rnn/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*J
valueAB? B9Expected shape for Tensor encoder_1/sequence_length:0 is 
s
"encoder_1/rnn/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
�
encoder_1/rnn/Assert/AssertAssertencoder_1/rnn/All"encoder_1/rnn/Assert/Assert/data_0encoder_1/rnn/stack"encoder_1/rnn/Assert/Assert/data_2encoder_1/rnn/Shape_1*
T
2*
	summarize
�
encoder_1/rnn/CheckSeqLenIdentityencoder_1/sequence_length^encoder_1/rnn/Assert/Assert*
T0*
_output_shapes	
:�
j
encoder_1/rnn/Shape_2Const*!
valueB"�   �   �   *
dtype0*
_output_shapes
:
m
#encoder_1/rnn/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: 
o
%encoder_1/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%encoder_1/rnn/strided_slice_2/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
�
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
#encoder_1/rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
o
%encoder_1/rnn/strided_slice_3/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
o
%encoder_1/rnn/strided_slice_3/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
�
encoder_1/rnn/strided_slice_3StridedSliceencoder_1/rnn/Shape_2#encoder_1/rnn/strided_slice_3/stack%encoder_1/rnn/strided_slice_3/stack_1%encoder_1/rnn/strided_slice_3/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
Z
encoder_1/rnn/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :�
�
encoder_1/rnn/stack_1Packencoder_1/rnn/strided_slice_3encoder_1/rnn/stack_1/1*
_output_shapes
:*
N*

axis *
T0
^
encoder_1/rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
encoder_1/rnn/zerosFillencoder_1/rnn/stack_1encoder_1/rnn/zeros/Const*(
_output_shapes
:����������*
T0
_
encoder_1/rnn/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
�
encoder_1/rnn/MinMinencoder_1/rnn/CheckSeqLenencoder_1/rnn/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
_
encoder_1/rnn/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
�
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
�
encoder_1/rnn/TensorArrayTensorArrayV3encoder_1/rnn/strided_slice_2*
dynamic_size( *
clear_after_read(*
_output_shapes

::*
element_shape:*
dtype0*9
tensor_array_name$"encoder_1/rnn/dynamic_rnn/output_0
�
encoder_1/rnn/TensorArray_1TensorArrayV3encoder_1/rnn/strided_slice_2*
_output_shapes

::*
dtype0*8
tensor_array_name#!encoder_1/rnn/dynamic_rnn/input_0*
dynamic_size( *
clear_after_read(*
element_shape:
{
&encoder_1/rnn/TensorArrayUnstack/ShapeConst*!
valueB"�   �   �   *
dtype0*
_output_shapes
:
~
4encoder_1/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
�
6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
.encoder_1/rnn/TensorArrayUnstack/strided_sliceStridedSlice&encoder_1/rnn/TensorArrayUnstack/Shape4encoder_1/rnn/TensorArrayUnstack/strided_slice/stack6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_16encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_2*
_output_shapes
: *
end_mask *
new_axis_mask *
ellipsis_mask *

begin_mask *
shrink_axis_mask*
T0*
Index0
n
,encoder_1/rnn/TensorArrayUnstack/range/startConst*
value	B : *
_output_shapes
: *
dtype0
n
,encoder_1/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
�
&encoder_1/rnn/TensorArrayUnstack/rangeRange,encoder_1/rnn/TensorArrayUnstack/range/start.encoder_1/rnn/TensorArrayUnstack/strided_slice,encoder_1/rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:���������
�
Hencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3encoder_1/rnn/TensorArray_1&encoder_1/rnn/TensorArrayUnstack/rangeencoder_1/transposeencoder_1/rnn/TensorArray_1:1*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: 
�
encoder_1/rnn/while/EnterEnterencoder_1/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
encoder_1/rnn/while/Enter_1Enterencoder_1/rnn/TensorArray:1*
is_constant( *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
�
encoder_1/rnn/while/Enter_2Enterencoder_1/initial_state_0_tiled*(
_output_shapes
:����������*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
�
encoder_1/rnn/while/Enter_3Enterencoder_1/initial_state_1_tiled*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:����������*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
encoder_1/rnn/while/Enter_4Enterencoder_1/initial_state_2_tiled*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:����������*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
encoder_1/rnn/while/Enter_5Enterencoder_1/initial_state_3_tiled*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:����������*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
encoder_1/rnn/while/MergeMergeencoder_1/rnn/while/Enter!encoder_1/rnn/while/NextIteration*
_output_shapes
: : *
T0*
N
�
encoder_1/rnn/while/Merge_1Mergeencoder_1/rnn/while/Enter_1#encoder_1/rnn/while/NextIteration_1*
N*
T0*
_output_shapes
:: 
�
encoder_1/rnn/while/Merge_2Mergeencoder_1/rnn/while/Enter_2#encoder_1/rnn/while/NextIteration_2*
T0*
N**
_output_shapes
:����������: 
�
encoder_1/rnn/while/Merge_3Mergeencoder_1/rnn/while/Enter_3#encoder_1/rnn/while/NextIteration_3*
N*
T0**
_output_shapes
:����������: 
�
encoder_1/rnn/while/Merge_4Mergeencoder_1/rnn/while/Enter_4#encoder_1/rnn/while/NextIteration_4*
T0*
N**
_output_shapes
:����������: 
�
encoder_1/rnn/while/Merge_5Mergeencoder_1/rnn/while/Enter_5#encoder_1/rnn/while/NextIteration_5**
_output_shapes
:����������: *
N*
T0
�
encoder_1/rnn/while/Less/EnterEnterencoder_1/rnn/strided_slice_2*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
|
encoder_1/rnn/while/LessLessencoder_1/rnn/while/Mergeencoder_1/rnn/while/Less/Enter*
_output_shapes
: *
T0
Z
encoder_1/rnn/while/LoopCondLoopCondencoder_1/rnn/while/Less*
_output_shapes
: 
�
encoder_1/rnn/while/SwitchSwitchencoder_1/rnn/while/Mergeencoder_1/rnn/while/LoopCond*
T0*,
_class"
 loc:@encoder_1/rnn/while/Merge*
_output_shapes
: : 
�
encoder_1/rnn/while/Switch_1Switchencoder_1/rnn/while/Merge_1encoder_1/rnn/while/LoopCond*
T0*.
_class$
" loc:@encoder_1/rnn/while/Merge_1*
_output_shapes

::
�
encoder_1/rnn/while/Switch_2Switchencoder_1/rnn/while/Merge_2encoder_1/rnn/while/LoopCond*
T0*.
_class$
" loc:@encoder_1/rnn/while/Merge_2*<
_output_shapes*
(:����������:����������
�
encoder_1/rnn/while/Switch_3Switchencoder_1/rnn/while/Merge_3encoder_1/rnn/while/LoopCond*
T0*<
_output_shapes*
(:����������:����������*.
_class$
" loc:@encoder_1/rnn/while/Merge_3
�
encoder_1/rnn/while/Switch_4Switchencoder_1/rnn/while/Merge_4encoder_1/rnn/while/LoopCond*<
_output_shapes*
(:����������:����������*.
_class$
" loc:@encoder_1/rnn/while/Merge_4*
T0
�
encoder_1/rnn/while/Switch_5Switchencoder_1/rnn/while/Merge_5encoder_1/rnn/while/LoopCond*.
_class$
" loc:@encoder_1/rnn/while/Merge_5*<
_output_shapes*
(:����������:����������*
T0
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
:����������
}
encoder_1/rnn/while/Identity_3Identityencoder_1/rnn/while/Switch_3:1*
T0*(
_output_shapes
:����������
}
encoder_1/rnn/while/Identity_4Identityencoder_1/rnn/while/Switch_4:1*(
_output_shapes
:����������*
T0
}
encoder_1/rnn/while/Identity_5Identityencoder_1/rnn/while/Switch_5:1*
T0*(
_output_shapes
:����������
�
+encoder_1/rnn/while/TensorArrayReadV3/EnterEnterencoder_1/rnn/TensorArray_1*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
-encoder_1/rnn/while/TensorArrayReadV3/Enter_1EnterHencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
%encoder_1/rnn/while/TensorArrayReadV3TensorArrayReadV3+encoder_1/rnn/while/TensorArrayReadV3/Enterencoder_1/rnn/while/Identity-encoder_1/rnn/while/TensorArrayReadV3/Enter_1*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
dtype0* 
_output_shapes
:
��
�
Tencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB"      *
_output_shapes
:*
dtype0
�
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/minConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB
 *
ף�*
dtype0*
_output_shapes
: 
�
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB
 *
ף=
�
\encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/RandomUniformRandomUniformTencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/shape*
seed2�*
T0*

seed{*
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
��
�
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/subSubRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/maxRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/min*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
_output_shapes
: *
T0
�
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/mulMul\encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/RandomUniformRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/sub*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
��*
T0
�
Nencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniformAddRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/mulRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/min*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
��*
T0
�
3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
VariableV2*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
��*
shape:
��*
dtype0*
shared_name *
	container 
�
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AssignAssign3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsNencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
validate_shape(* 
_output_shapes
:
��
�
8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/readIdentity3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0* 
_output_shapes
:
��
�
Iencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axisConst^encoder_1/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
�
Dencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concatConcatV2%encoder_1/rnn/while/TensorArrayReadV3encoder_1/rnn/while/Identity_3Iencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*

Tidx0*
T0*
N* 
_output_shapes
:
��
�
Jencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read* 
_output_shapes
:
��*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
Dencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMulMatMulDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concatJencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a( 
�
Dencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Initializer/ConstConst*
_output_shapes	
:�*
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
valueB�*    
�
2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
VariableV2*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AssignAssign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesDencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Initializer/Const*
_output_shapes	
:�*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
T0*
use_locking(
�
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/readIdentity2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:�*
T0
�
Aencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/EnterEnter7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read*
_output_shapes	
:�*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAddBiasAddDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMulAencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC* 
_output_shapes
:
��
�
Cencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dimConst^encoder_1/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
�
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/splitSplitCencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split*
T0
�
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add/yConst^encoder_1/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/addAdd;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add/y*
T0* 
_output_shapes
:
��
�
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/SigmoidSigmoid7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add*
T0* 
_output_shapes
:
��
�
7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mulMul;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoidencoder_1/rnn/while/Identity_2*
T0* 
_output_shapes
:
��
�
=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1Sigmoid9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split* 
_output_shapes
:
��*
T0
�
8encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/TanhTanh;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:1*
T0* 
_output_shapes
:
��
�
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1Mul=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_18encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh* 
_output_shapes
:
��*
T0
�
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1Add7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1* 
_output_shapes
:
��*
T0
�
=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2Sigmoid;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:3* 
_output_shapes
:
��*
T0
�
:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1Tanh9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1* 
_output_shapes
:
��*
T0
�
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2Mul=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0* 
_output_shapes
:
��
�
Tencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB"      
�
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB
 *
ף�
�
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/maxConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB
 *
ף=*
dtype0*
_output_shapes
: 
�
\encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/RandomUniformRandomUniformTencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/shape* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*

seed{*
T0*
seed2�
�
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/subSubRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/maxRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
�
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/mulMul\encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/RandomUniformRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
��
�
Nencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniformAddRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/mulRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
��
�
3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
VariableV2*
shared_name *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AssignAssign3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsNencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform* 
_output_shapes
:
��*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0*
use_locking(
�
8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/readIdentity3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0* 
_output_shapes
:
��
�
Iencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axisConst^encoder_1/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
�
Dencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concatConcatV29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2encoder_1/rnn/while/Identity_5Iencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis* 
_output_shapes
:
��*
T0*

Tidx0*
N
�
Jencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read* 
_output_shapes
:
��*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
Dencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMulMatMulDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concatJencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter*
transpose_b( * 
_output_shapes
:
��*
transpose_a( *
T0
�
Dencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Initializer/ConstConst*
_output_shapes	
:�*
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
valueB�*    
�
2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
VariableV2*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:�*
shape:�*
dtype0*
shared_name *
	container 
�
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AssignAssign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesDencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Initializer/Const*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
validate_shape(*
_output_shapes	
:�
�
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/readIdentity2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:�*
T0
�
Aencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/EnterEnter7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/read*
parallel_iterations *
T0*
_output_shapes	
:�*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(
�
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAddBiasAddDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMulAencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter* 
_output_shapes
:
��*
data_formatNHWC*
T0
�
Cencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dimConst^encoder_1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
�
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/splitSplitCencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd*
T0*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split
�
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add/yConst^encoder_1/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/addAdd;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:29encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add/y*
T0* 
_output_shapes
:
��
�
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/SigmoidSigmoid7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add* 
_output_shapes
:
��*
T0
�
7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mulMul;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoidencoder_1/rnn/while/Identity_4*
T0* 
_output_shapes
:
��
�
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1Sigmoid9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split*
T0* 
_output_shapes
:
��
�
8encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/TanhTanh;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:1* 
_output_shapes
:
��*
T0
�
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1Mul=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_18encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0* 
_output_shapes
:
��
�
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1Add7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1* 
_output_shapes
:
��*
T0
�
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2Sigmoid;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:3*
T0* 
_output_shapes
:
��
�
:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1Tanh9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1*
T0* 
_output_shapes
:
��
�
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2Mul=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1* 
_output_shapes
:
��*
T0
�
&encoder_1/rnn/while/GreaterEqual/EnterEnterencoder_1/rnn/CheckSeqLen*
is_constant(*
_output_shapes	
:�*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
�
 encoder_1/rnn/while/GreaterEqualGreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:�
�
 encoder_1/rnn/while/Select/EnterEnterencoder_1/rnn/zeros*(
_output_shapes
:����������*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
encoder_1/rnn/while/SelectSelect encoder_1/rnn/while/GreaterEqual encoder_1/rnn/while/Select/Enter9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0* 
_output_shapes
:
��
�
"encoder_1/rnn/while/GreaterEqual_1GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:�
�
encoder_1/rnn/while/Select_1Select"encoder_1/rnn/while/GreaterEqual_1encoder_1/rnn/while/Identity_29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1*
T0* 
_output_shapes
:
��
�
"encoder_1/rnn/while/GreaterEqual_2GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:�
�
encoder_1/rnn/while/Select_2Select"encoder_1/rnn/while/GreaterEqual_2encoder_1/rnn/while/Identity_39encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2* 
_output_shapes
:
��*
T0
�
"encoder_1/rnn/while/GreaterEqual_3GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
_output_shapes	
:�*
T0
�
encoder_1/rnn/while/Select_3Select"encoder_1/rnn/while/GreaterEqual_3encoder_1/rnn/while/Identity_49encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1* 
_output_shapes
:
��*
T0
�
"encoder_1/rnn/while/GreaterEqual_4GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:�
�
encoder_1/rnn/while/Select_4Select"encoder_1/rnn/while/GreaterEqual_4encoder_1/rnn/while/Identity_59encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0* 
_output_shapes
:
��
�
=encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterencoder_1/rnn/TensorArray*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *,
_class"
 loc:@encoder_1/rnn/TensorArray*
T0
�
7encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3=encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterencoder_1/rnn/while/Identityencoder_1/rnn/while/Selectencoder_1/rnn/while/Identity_1*
T0*
_output_shapes
: *,
_class"
 loc:@encoder_1/rnn/TensorArray
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
!encoder_1/rnn/while/NextIterationNextIterationencoder_1/rnn/while/add*
T0*
_output_shapes
: 
�
#encoder_1/rnn/while/NextIteration_1NextIteration7encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
}
#encoder_1/rnn/while/NextIteration_2NextIterationencoder_1/rnn/while/Select_1* 
_output_shapes
:
��*
T0
}
#encoder_1/rnn/while/NextIteration_3NextIterationencoder_1/rnn/while/Select_2*
T0* 
_output_shapes
:
��
}
#encoder_1/rnn/while/NextIteration_4NextIterationencoder_1/rnn/while/Select_3*
T0* 
_output_shapes
:
��
}
#encoder_1/rnn/while/NextIteration_5NextIterationencoder_1/rnn/while/Select_4*
T0* 
_output_shapes
:
��
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
encoder_1/rnn/while/Exit_2Exitencoder_1/rnn/while/Switch_2*(
_output_shapes
:����������*
T0
s
encoder_1/rnn/while/Exit_3Exitencoder_1/rnn/while/Switch_3*(
_output_shapes
:����������*
T0
s
encoder_1/rnn/while/Exit_4Exitencoder_1/rnn/while/Switch_4*
T0*(
_output_shapes
:����������
s
encoder_1/rnn/while/Exit_5Exitencoder_1/rnn/while/Switch_5*(
_output_shapes
:����������*
T0
�
0encoder_1/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3encoder_1/rnn/TensorArrayencoder_1/rnn/while/Exit_1*
_output_shapes
: *,
_class"
 loc:@encoder_1/rnn/TensorArray
�
*encoder_1/rnn/TensorArrayStack/range/startConst*
_output_shapes
: *
dtype0*
value	B : *,
_class"
 loc:@encoder_1/rnn/TensorArray
�
*encoder_1/rnn/TensorArrayStack/range/deltaConst*
value	B :*,
_class"
 loc:@encoder_1/rnn/TensorArray*
_output_shapes
: *
dtype0
�
$encoder_1/rnn/TensorArrayStack/rangeRange*encoder_1/rnn/TensorArrayStack/range/start0encoder_1/rnn/TensorArrayStack/TensorArraySizeV3*encoder_1/rnn/TensorArrayStack/range/delta*

Tidx0*,
_class"
 loc:@encoder_1/rnn/TensorArray*#
_output_shapes
:���������
�
2encoder_1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3encoder_1/rnn/TensorArray$encoder_1/rnn/TensorArrayStack/rangeencoder_1/rnn/while/Exit_1*,
_class"
 loc:@encoder_1/rnn/TensorArray*
element_shape:
��*%
_output_shapes
:���*
dtype0
q
encoder_1/rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
�
encoder_1/rnn/transpose	Transpose2encoder_1/rnn/TensorArrayStack/TensorArrayGatherV3encoder_1/rnn/transpose/perm*
Tperm0*%
_output_shapes
:���*
T0
�
6output_projection/W/Initializer/truncated_normal/shapeConst*&
_class
loc:@output_projection/W*
valueB"�      *
dtype0*
_output_shapes
:
�
5output_projection/W/Initializer/truncated_normal/meanConst*&
_class
loc:@output_projection/W*
valueB
 *    *
_output_shapes
: *
dtype0
�
7output_projection/W/Initializer/truncated_normal/stddevConst*&
_class
loc:@output_projection/W*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
@output_projection/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6output_projection/W/Initializer/truncated_normal/shape*&
_class
loc:@output_projection/W*
_output_shapes
:	�*
T0*
dtype0*
seed2�*

seed{
�
4output_projection/W/Initializer/truncated_normal/mulMul@output_projection/W/Initializer/truncated_normal/TruncatedNormal7output_projection/W/Initializer/truncated_normal/stddev*&
_class
loc:@output_projection/W*
_output_shapes
:	�*
T0
�
0output_projection/W/Initializer/truncated_normalAdd4output_projection/W/Initializer/truncated_normal/mul5output_projection/W/Initializer/truncated_normal/mean*
T0*&
_class
loc:@output_projection/W*
_output_shapes
:	�
�
output_projection/W
VariableV2*
_output_shapes
:	�*
dtype0*
shape:	�*
	container *&
_class
loc:@output_projection/W*
shared_name 
�
output_projection/W/AssignAssignoutput_projection/W0output_projection/W/Initializer/truncated_normal*&
_class
loc:@output_projection/W*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
output_projection/W/readIdentityoutput_projection/W*&
_class
loc:@output_projection/W*
_output_shapes
:	�*
T0
�
%output_projection/b/Initializer/ConstConst*&
_class
loc:@output_projection/b*
valueB*���=*
_output_shapes
:*
dtype0
�
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
�
output_projection/b/AssignAssignoutput_projection/b%output_projection/b/Initializer/Const*
use_locking(*
T0*&
_class
loc:@output_projection/b*
validate_shape(*
_output_shapes
:
�
output_projection/b/readIdentityoutput_projection/b*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0
�
"output_projection/xw_plus_b/MatMulMatMulencoder_1/rnn/while/Exit_4output_projection/W/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
output_projection/xw_plus_bBiasAdd"output_projection/xw_plus_b/MatMuloutput_projection/b/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
s
output_projection/SoftmaxSoftmaxoutput_projection/xw_plus_b*'
_output_shapes
:���������*
T0
d
"output_projection/ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
�
output_projection/ArgMaxArgMaxoutput_projection/xw_plus_b"output_projection/ArgMax/dimension*#
_output_shapes
:���������*
T0*

Tidx0
o

loss/ConstConst*
_output_shapes
:*
dtype0*1
value(B&"  �?��z?`�p?bX?�p}?  �?m�{>
j
loss/MulMuloutput_projection/xw_plus_b
loss/Const*
T0*'
_output_shapes
:���������
X
	loss/CastCastcond/Merge_1*
_output_shapes
:	�*

DstT0*

SrcT0
]
loss/logistic_loss/sub/yConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
h
loss/logistic_loss/subSub
loss/Constloss/logistic_loss/sub/y*
T0*
_output_shapes
:
j
loss/logistic_loss/mulMulloss/logistic_loss/sub	loss/Cast*
_output_shapes
:	�*
T0
]
loss/logistic_loss/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
y
loss/logistic_loss/addAddloss/logistic_loss/add/xloss/logistic_loss/mul*
T0*
_output_shapes
:	�
_
loss/logistic_loss/sub_1/xConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
p
loss/logistic_loss/sub_1Subloss/logistic_loss/sub_1/x	loss/Cast*
_output_shapes
:	�*
T0
m
loss/logistic_loss/mul_1Mulloss/logistic_loss/sub_1loss/Mul*
T0*
_output_shapes
:	�
Y
loss/logistic_loss/AbsAbsloss/Mul*'
_output_shapes
:���������*
T0
g
loss/logistic_loss/NegNegloss/logistic_loss/Abs*
T0*'
_output_shapes
:���������
g
loss/logistic_loss/ExpExploss/logistic_loss/Neg*
T0*'
_output_shapes
:���������
k
loss/logistic_loss/Log1pLog1ploss/logistic_loss/Exp*'
_output_shapes
:���������*
T0
[
loss/logistic_loss/Neg_1Negloss/Mul*'
_output_shapes
:���������*
T0
k
loss/logistic_loss/ReluReluloss/logistic_loss/Neg_1*'
_output_shapes
:���������*
T0
�
loss/logistic_loss/add_1Addloss/logistic_loss/Log1ploss/logistic_loss/Relu*
T0*'
_output_shapes
:���������
{
loss/logistic_loss/mul_2Mulloss/logistic_loss/addloss/logistic_loss/add_1*
T0*
_output_shapes
:	�
w
loss/logistic_lossAddloss/logistic_loss/mul_1loss/logistic_loss/mul_2*
T0*
_output_shapes
:	�
]
loss/Const_1Const*
dtype0*
_output_shapes
:*
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
accuracy/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
t
accuracy/ArgMaxArgMaxcond/Merge_1accuracy/ArgMax/dimension*

Tidx0*
T0*
_output_shapes	
:�
h
accuracy/EqualEqualoutput_projection/ArgMaxaccuracy/ArgMax*
_output_shapes	
:�*
T0	
Z
accuracy/CastCastaccuracy/Equal*

SrcT0
*
_output_shapes	
:�*

DstT0
X
accuracy/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
v
accuracy/accuracyMeanaccuracy/Castaccuracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
`
learning_rate/learning_rateConst*
valueB
 *�Q8*
_output_shapes
: *
dtype0
Y
learning_rate/CastCastVariable/read*

SrcT0*
_output_shapes
: *

DstT0
Y
learning_rate/Cast_1/xConst*
value
B :�'*
_output_shapes
: *
dtype0
d
learning_rate/Cast_1Castlearning_rate/Cast_1/x*

SrcT0*
_output_shapes
: *

DstT0
[
learning_rate/Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *��u?
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
learning_rateMullearning_rate/learning_ratelearning_rate/Pow*
_output_shapes
: *
T0
`
gradients/ShapeConst*
valueB"�      *
dtype0*
_output_shapes
:
T
gradients/ConstConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
b
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
:	�*
T0
S
gradients/f_countConst*
value	B : *
_output_shapes
: *
dtype0
�
gradients/f_count_1Entergradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
_output_shapes
: : *
T0*
N
l
gradients/SwitchSwitchgradients/Mergeencoder_1/rnn/while/LoopCond*
_output_shapes
: : *
T0
p
gradients/Add/yConst^encoder_1/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
T0*
_output_shapes
: 
�
gradients/NextIterationNextIterationgradients/AddA^gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPushA^gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPushA^gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPushA^gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPushZ^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPushb^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPushe^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPushZ^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPushb^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPushe^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPushc^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPush*
_output_shapes
: *
T0
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
�
gradients/b_count_1Entergradients/f_count_2*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
N*
T0*
_output_shapes
: : 
�
gradients/GreaterEqual/EnterEntergradients/b_count*
is_constant(*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
x
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
O
gradients/b_count_2LoopCondgradients/GreaterEqual*
_output_shapes
: 
g
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
_output_shapes
: : *
T0
i
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
�
gradients/NextIteration_1NextIterationgradients/Sub>^gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/b_sync*
T0*
_output_shapes
: 
P
gradients/b_count_3Exitgradients/Switch_1*
T0*
_output_shapes
: 
x
'gradients/loss/logistic_loss_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"�      
z
)gradients/loss/logistic_loss_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"�      
�
7gradients/loss/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/loss/logistic_loss_grad/Shape)gradients/loss/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
%gradients/loss/logistic_loss_grad/SumSumgradients/Fill7gradients/loss/logistic_loss_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
)gradients/loss/logistic_loss_grad/ReshapeReshape%gradients/loss/logistic_loss_grad/Sum'gradients/loss/logistic_loss_grad/Shape*
Tshape0*
_output_shapes
:	�*
T0
�
'gradients/loss/logistic_loss_grad/Sum_1Sumgradients/Fill9gradients/loss/logistic_loss_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
+gradients/loss/logistic_loss_grad/Reshape_1Reshape'gradients/loss/logistic_loss_grad/Sum_1)gradients/loss/logistic_loss_grad/Shape_1*
_output_shapes
:	�*
Tshape0*
T0
~
-gradients/loss/logistic_loss/mul_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"�      
w
/gradients/loss/logistic_loss/mul_1_grad/Shape_1Shapeloss/Mul*
out_type0*
_output_shapes
:*
T0
�
=gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/mul_1_grad/Shape/gradients/loss/logistic_loss/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
+gradients/loss/logistic_loss/mul_1_grad/mulMul)gradients/loss/logistic_loss_grad/Reshapeloss/Mul*
T0*
_output_shapes
:	�
�
+gradients/loss/logistic_loss/mul_1_grad/SumSum+gradients/loss/logistic_loss/mul_1_grad/mul=gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
/gradients/loss/logistic_loss/mul_1_grad/ReshapeReshape+gradients/loss/logistic_loss/mul_1_grad/Sum-gradients/loss/logistic_loss/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
�
-gradients/loss/logistic_loss/mul_1_grad/mul_1Mulloss/logistic_loss/sub_1)gradients/loss/logistic_loss_grad/Reshape*
_output_shapes
:	�*
T0
�
-gradients/loss/logistic_loss/mul_1_grad/Sum_1Sum-gradients/loss/logistic_loss/mul_1_grad/mul_1?gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
1gradients/loss/logistic_loss/mul_1_grad/Reshape_1Reshape-gradients/loss/logistic_loss/mul_1_grad/Sum_1/gradients/loss/logistic_loss/mul_1_grad/Shape_1*
T0*'
_output_shapes
:���������*
Tshape0
~
-gradients/loss/logistic_loss/mul_2_grad/ShapeConst*
valueB"�      *
dtype0*
_output_shapes
:
�
/gradients/loss/logistic_loss/mul_2_grad/Shape_1Shapeloss/logistic_loss/add_1*
out_type0*
_output_shapes
:*
T0
�
=gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/mul_2_grad/Shape/gradients/loss/logistic_loss/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
+gradients/loss/logistic_loss/mul_2_grad/mulMul+gradients/loss/logistic_loss_grad/Reshape_1loss/logistic_loss/add_1*
T0*
_output_shapes
:	�
�
+gradients/loss/logistic_loss/mul_2_grad/SumSum+gradients/loss/logistic_loss/mul_2_grad/mul=gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
/gradients/loss/logistic_loss/mul_2_grad/ReshapeReshape+gradients/loss/logistic_loss/mul_2_grad/Sum-gradients/loss/logistic_loss/mul_2_grad/Shape*
_output_shapes
:	�*
Tshape0*
T0
�
-gradients/loss/logistic_loss/mul_2_grad/mul_1Mulloss/logistic_loss/add+gradients/loss/logistic_loss_grad/Reshape_1*
_output_shapes
:	�*
T0
�
-gradients/loss/logistic_loss/mul_2_grad/Sum_1Sum-gradients/loss/logistic_loss/mul_2_grad/mul_1?gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
1gradients/loss/logistic_loss/mul_2_grad/Reshape_1Reshape-gradients/loss/logistic_loss/mul_2_grad/Sum_1/gradients/loss/logistic_loss/mul_2_grad/Shape_1*
T0*'
_output_shapes
:���������*
Tshape0
�
-gradients/loss/logistic_loss/add_1_grad/ShapeShapeloss/logistic_loss/Log1p*
_output_shapes
:*
out_type0*
T0
�
/gradients/loss/logistic_loss/add_1_grad/Shape_1Shapeloss/logistic_loss/Relu*
out_type0*
_output_shapes
:*
T0
�
=gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/add_1_grad/Shape/gradients/loss/logistic_loss/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
+gradients/loss/logistic_loss/add_1_grad/SumSum1gradients/loss/logistic_loss/mul_2_grad/Reshape_1=gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
/gradients/loss/logistic_loss/add_1_grad/ReshapeReshape+gradients/loss/logistic_loss/add_1_grad/Sum-gradients/loss/logistic_loss/add_1_grad/Shape*
T0*'
_output_shapes
:���������*
Tshape0
�
-gradients/loss/logistic_loss/add_1_grad/Sum_1Sum1gradients/loss/logistic_loss/mul_2_grad/Reshape_1?gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
1gradients/loss/logistic_loss/add_1_grad/Reshape_1Reshape-gradients/loss/logistic_loss/add_1_grad/Sum_1/gradients/loss/logistic_loss/add_1_grad/Shape_1*
T0*'
_output_shapes
:���������*
Tshape0
�
-gradients/loss/logistic_loss/Log1p_grad/add/xConst0^gradients/loss/logistic_loss/add_1_grad/Reshape*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
+gradients/loss/logistic_loss/Log1p_grad/addAdd-gradients/loss/logistic_loss/Log1p_grad/add/xloss/logistic_loss/Exp*'
_output_shapes
:���������*
T0
�
2gradients/loss/logistic_loss/Log1p_grad/Reciprocal
Reciprocal+gradients/loss/logistic_loss/Log1p_grad/add*
T0*'
_output_shapes
:���������
�
+gradients/loss/logistic_loss/Log1p_grad/mulMul/gradients/loss/logistic_loss/add_1_grad/Reshape2gradients/loss/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:���������*
T0
�
/gradients/loss/logistic_loss/Relu_grad/ReluGradReluGrad1gradients/loss/logistic_loss/add_1_grad/Reshape_1loss/logistic_loss/Relu*
T0*'
_output_shapes
:���������
�
)gradients/loss/logistic_loss/Exp_grad/mulMul+gradients/loss/logistic_loss/Log1p_grad/mulloss/logistic_loss/Exp*'
_output_shapes
:���������*
T0
�
+gradients/loss/logistic_loss/Neg_1_grad/NegNeg/gradients/loss/logistic_loss/Relu_grad/ReluGrad*'
_output_shapes
:���������*
T0
�
)gradients/loss/logistic_loss/Neg_grad/NegNeg)gradients/loss/logistic_loss/Exp_grad/mul*'
_output_shapes
:���������*
T0
n
*gradients/loss/logistic_loss/Abs_grad/SignSignloss/Mul*'
_output_shapes
:���������*
T0
�
)gradients/loss/logistic_loss/Abs_grad/mulMul)gradients/loss/logistic_loss/Neg_grad/Neg*gradients/loss/logistic_loss/Abs_grad/Sign*'
_output_shapes
:���������*
T0
�
gradients/AddNAddN1gradients/loss/logistic_loss/mul_1_grad/Reshape_1+gradients/loss/logistic_loss/Neg_1_grad/Neg)gradients/loss/logistic_loss/Abs_grad/mul*D
_class:
86loc:@gradients/loss/logistic_loss/mul_1_grad/Reshape_1*'
_output_shapes
:���������*
T0*
N
x
gradients/loss/Mul_grad/ShapeShapeoutput_projection/xw_plus_b*
T0*
_output_shapes
:*
out_type0
i
gradients/loss/Mul_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
-gradients/loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/Mul_grad/Shapegradients/loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
p
gradients/loss/Mul_grad/mulMulgradients/AddN
loss/Const*'
_output_shapes
:���������*
T0
�
gradients/loss/Mul_grad/SumSumgradients/loss/Mul_grad/mul-gradients/loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/loss/Mul_grad/ReshapeReshapegradients/loss/Mul_grad/Sumgradients/loss/Mul_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/loss/Mul_grad/mul_1Muloutput_projection/xw_plus_bgradients/AddN*'
_output_shapes
:���������*
T0
�
gradients/loss/Mul_grad/Sum_1Sumgradients/loss/Mul_grad/mul_1/gradients/loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
!gradients/loss/Mul_grad/Reshape_1Reshapegradients/loss/Mul_grad/Sum_1gradients/loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
�
6gradients/output_projection/xw_plus_b_grad/BiasAddGradBiasAddGradgradients/loss/Mul_grad/Reshape*
_output_shapes
:*
data_formatNHWC*
T0
�
8gradients/output_projection/xw_plus_b/MatMul_grad/MatMulMatMulgradients/loss/Mul_grad/Reshapeoutput_projection/W/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1MatMulencoder_1/rnn/while/Exit_4gradients/loss/Mul_grad/Reshape*
transpose_b( *
_output_shapes
:	�*
transpose_a(*
T0
`
gradients/zeros_like	ZerosLikeencoder_1/rnn/while/Exit_1*
_output_shapes
:*
T0
r
gradients/zeros_like_1	ZerosLikeencoder_1/rnn/while/Exit_2*(
_output_shapes
:����������*
T0
r
gradients/zeros_like_2	ZerosLikeencoder_1/rnn/while/Exit_3*
T0*(
_output_shapes
:����������
r
gradients/zeros_like_3	ZerosLikeencoder_1/rnn/while/Exit_5*(
_output_shapes
:����������*
T0
�
0gradients/encoder_1/rnn/while/Exit_4_grad/b_exitEnter8gradients/output_projection/xw_plus_b/MatMul_grad/MatMul*
is_constant( *(
_output_shapes
:����������*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
�
0gradients/encoder_1/rnn/while/Exit_1_grad/b_exitEntergradients/zeros_like*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
�
0gradients/encoder_1/rnn/while/Exit_2_grad/b_exitEntergradients/zeros_like_1*
parallel_iterations *
T0*(
_output_shapes
:����������*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
�
0gradients/encoder_1/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_like_2*
is_constant( *(
_output_shapes
:����������*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
�
0gradients/encoder_1/rnn/while/Exit_5_grad/b_exitEntergradients/zeros_like_3*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:����������*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
4gradients/encoder_1/rnn/while/Switch_4_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_4_grad/b_exit;gradients/encoder_1/rnn/while/Switch_4_grad_1/NextIteration**
_output_shapes
:����������: *
T0*
N
�
4gradients/encoder_1/rnn/while/Switch_2_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_2_grad/b_exit;gradients/encoder_1/rnn/while/Switch_2_grad_1/NextIteration**
_output_shapes
:����������: *
T0*
N
�
4gradients/encoder_1/rnn/while/Switch_3_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_3_grad/b_exit;gradients/encoder_1/rnn/while/Switch_3_grad_1/NextIteration*
N*
T0**
_output_shapes
:����������: 
�
4gradients/encoder_1/rnn/while/Switch_5_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_5_grad/b_exit;gradients/encoder_1/rnn/while/Switch_5_grad_1/NextIteration*
T0*
N**
_output_shapes
:����������: 
�
1gradients/encoder_1/rnn/while/Merge_4_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_4_grad/b_switch*4
_output_shapes"
 :����������:
��*
T0
�
1gradients/encoder_1/rnn/while/Merge_2_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*4
_output_shapes"
 :����������:
��*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_2_grad/b_switch*
T0
�
1gradients/encoder_1/rnn/while/Merge_3_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*
T0*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_3_grad/b_switch*4
_output_shapes"
 :����������:
��
�
1gradients/encoder_1/rnn/while/Merge_5_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_5_grad/b_switchgradients/b_count_2*
T0*4
_output_shapes"
 :����������:
��*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_5_grad/b_switch
�
/gradients/encoder_1/rnn/while/Enter_4_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_4_grad/Switch*(
_output_shapes
:����������*
T0
�
/gradients/encoder_1/rnn/while/Enter_2_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_2_grad/Switch*(
_output_shapes
:����������*
T0
�
/gradients/encoder_1/rnn/while/Enter_3_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_3_grad/Switch*(
_output_shapes
:����������*
T0
�
/gradients/encoder_1/rnn/while/Enter_5_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_5_grad/Switch*(
_output_shapes
:����������*
T0
�
4gradients/encoder_1/initial_state_2_tiled_grad/ShapeConst*
valueB"   �   *
dtype0*
_output_shapes
:
�
4gradients/encoder_1/initial_state_2_tiled_grad/stackPack)encoder_1/initial_state_2_tiled/multiples4gradients/encoder_1/initial_state_2_tiled_grad/Shape*

axis *
_output_shapes

:*
T0*
N
�
=gradients/encoder_1/initial_state_2_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_2_tiled_grad/stack*
_output_shapes
: *
T0
�
>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub/yConst*
dtype0*
_output_shapes
: *
value	B :
�
<gradients/encoder_1/initial_state_2_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_2_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub/y*
_output_shapes
: *
T0
�
Dgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 
�
Dgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
>gradients/encoder_1/initial_state_2_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_2_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/delta*

Tidx0*
_output_shapes
:
�
>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_2_tiled_grad/transpose/Range*
T0*
_output_shapes
:
�
8gradients/encoder_1/initial_state_2_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_2_tiled_grad/stack>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub_1*
Tperm0*
T0*
_output_shapes

:
�
<gradients/encoder_1/initial_state_2_tiled_grad/Reshape/shapeConst*
valueB:
���������*
_output_shapes
:*
dtype0
�
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
value	B : *
_output_shapes
: *
dtype0
|
:gradients/encoder_1/initial_state_2_tiled_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
�
4gradients/encoder_1/initial_state_2_tiled_grad/rangeRange:gradients/encoder_1/initial_state_2_tiled_grad/range/start3gradients/encoder_1/initial_state_2_tiled_grad/Size:gradients/encoder_1/initial_state_2_tiled_grad/range/delta*
_output_shapes
:*

Tidx0
�
8gradients/encoder_1/initial_state_2_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_4_grad/Exit6gradients/encoder_1/initial_state_2_tiled_grad/Reshape*
Tshape0*J
_output_shapes8
6:4������������������������������������*
T0
�
2gradients/encoder_1/initial_state_2_tiled_grad/SumSum8gradients/encoder_1/initial_state_2_tiled_grad/Reshape_14gradients/encoder_1/initial_state_2_tiled_grad/range*
_output_shapes
:	�*
T0*
	keep_dims( *

Tidx0
�
<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4
�
?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
T0
�
@gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_4^gradients/Add*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
_output_shapes
:*
swap_memory( *
T0
�
Hgradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4
�
?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*(
_output_shapes
:����������*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4
�
=gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/b_syncControlTrigger@^gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopa^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopd^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopa^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopd^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopb^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop
�
6gradients/encoder_1/rnn/while/Select_3_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop*
T0*(
_output_shapes
:����������
�
8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*

stack_name *
_output_shapes
:
�
;gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
<gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_3^gradients/Add*
T0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
_output_shapes
:*
swap_memory( 
�
Dgradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc*
T0*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
;gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
_output_shapes	
:�
�
2gradients/encoder_1/rnn/while/Select_3_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_4_grad/Switch:16gradients/encoder_1/rnn/while/Select_3_grad/zeros_like* 
_output_shapes
:
��*
T0
�
4gradients/encoder_1/rnn/while/Select_3_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_3_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_4_grad/Switch:1*
T0* 
_output_shapes
:
��
�
4gradients/encoder_1/initial_state_0_tiled_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   �   
�
4gradients/encoder_1/initial_state_0_tiled_grad/stackPack)encoder_1/initial_state_0_tiled/multiples4gradients/encoder_1/initial_state_0_tiled_grad/Shape*
T0*

axis *
N*
_output_shapes

:
�
=gradients/encoder_1/initial_state_0_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_0_tiled_grad/stack*
T0*
_output_shapes
: 
�
>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub/yConst*
value	B :*
_output_shapes
: *
dtype0
�
<gradients/encoder_1/initial_state_0_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_0_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub/y*
_output_shapes
: *
T0
�
Dgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/startConst*
value	B : *
_output_shapes
: *
dtype0
�
Dgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
�
>gradients/encoder_1/initial_state_0_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_0_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/delta*

Tidx0*
_output_shapes
:
�
>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_0_tiled_grad/transpose/Range*
_output_shapes
:*
T0
�
8gradients/encoder_1/initial_state_0_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_0_tiled_grad/stack>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub_1*
Tperm0*
_output_shapes

:*
T0
�
<gradients/encoder_1/initial_state_0_tiled_grad/Reshape/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
6gradients/encoder_1/initial_state_0_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_0_tiled_grad/transpose<gradients/encoder_1/initial_state_0_tiled_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
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
:gradients/encoder_1/initial_state_0_tiled_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
�
4gradients/encoder_1/initial_state_0_tiled_grad/rangeRange:gradients/encoder_1/initial_state_0_tiled_grad/range/start3gradients/encoder_1/initial_state_0_tiled_grad/Size:gradients/encoder_1/initial_state_0_tiled_grad/range/delta*
_output_shapes
:*

Tidx0
�
8gradients/encoder_1/initial_state_0_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_2_grad/Exit6gradients/encoder_1/initial_state_0_tiled_grad/Reshape*
Tshape0*J
_output_shapes8
6:4������������������������������������*
T0
�
2gradients/encoder_1/initial_state_0_tiled_grad/SumSum8gradients/encoder_1/initial_state_0_tiled_grad/Reshape_14gradients/encoder_1/initial_state_0_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	�
�
<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *1
_class'
%#loc:@encoder_1/rnn/while/Identity_2
�
?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*
T0*
is_constant(
�
@gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_2^gradients/Add*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*
_output_shapes
:*
swap_memory( 
�
Hgradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*(
_output_shapes
:����������*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2
�
6gradients/encoder_1/rnn/while/Select_1_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop*(
_output_shapes
:����������*
T0
�
8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_accStack*
	elem_type0
*

stack_name *
_output_shapes
:*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1
�
;gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc*
is_constant(*
T0*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
�
<gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_1^gradients/Add*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
_output_shapes
:*
swap_memory( *
T0

�
Dgradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*
_output_shapes	
:�*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1
�
2gradients/encoder_1/rnn/while/Select_1_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_2_grad/Switch:16gradients/encoder_1/rnn/while/Select_1_grad/zeros_like* 
_output_shapes
:
��*
T0
�
4gradients/encoder_1/rnn/while/Select_1_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_1_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_2_grad/Switch:1*
T0* 
_output_shapes
:
��
�
4gradients/encoder_1/initial_state_1_tiled_grad/ShapeConst*
valueB"   �   *
dtype0*
_output_shapes
:
�
4gradients/encoder_1/initial_state_1_tiled_grad/stackPack)encoder_1/initial_state_1_tiled/multiples4gradients/encoder_1/initial_state_1_tiled_grad/Shape*
_output_shapes

:*
N*

axis *
T0
�
=gradients/encoder_1/initial_state_1_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_1_tiled_grad/stack*
T0*
_output_shapes
: 
�
>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub/yConst*
value	B :*
_output_shapes
: *
dtype0
�
<gradients/encoder_1/initial_state_1_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_1_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub/y*
_output_shapes
: *
T0
�
Dgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/startConst*
value	B : *
_output_shapes
: *
dtype0
�
Dgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
>gradients/encoder_1/initial_state_1_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_1_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/delta*

Tidx0*
_output_shapes
:
�
>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_1_tiled_grad/transpose/Range*
T0*
_output_shapes
:
�
8gradients/encoder_1/initial_state_1_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_1_tiled_grad/stack>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub_1*
Tperm0*
_output_shapes

:*
T0
�
<gradients/encoder_1/initial_state_1_tiled_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
6gradients/encoder_1/initial_state_1_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_1_tiled_grad/transpose<gradients/encoder_1/initial_state_1_tiled_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
u
3gradients/encoder_1/initial_state_1_tiled_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
|
:gradients/encoder_1/initial_state_1_tiled_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
|
:gradients/encoder_1/initial_state_1_tiled_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
4gradients/encoder_1/initial_state_1_tiled_grad/rangeRange:gradients/encoder_1/initial_state_1_tiled_grad/range/start3gradients/encoder_1/initial_state_1_tiled_grad/Size:gradients/encoder_1/initial_state_1_tiled_grad/range/delta*

Tidx0*
_output_shapes
:
�
8gradients/encoder_1/initial_state_1_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_3_grad/Exit6gradients/encoder_1/initial_state_1_tiled_grad/Reshape*J
_output_shapes8
6:4������������������������������������*
Tshape0*
T0
�
2gradients/encoder_1/initial_state_1_tiled_grad/SumSum8gradients/encoder_1/initial_state_1_tiled_grad/Reshape_14gradients/encoder_1/initial_state_1_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	�
�
<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_accStack*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*
_output_shapes
:*

stack_name 
�
?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*
T0*
is_constant(
�
@gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_3^gradients/Add*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*
_output_shapes
:*
swap_memory( 
�
Hgradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*(
_output_shapes
:����������
�
6gradients/encoder_1/rnn/while/Select_2_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop*(
_output_shapes
:����������*
T0
�
8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_accStack*
	elem_type0
*
_output_shapes
:*

stack_name *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2
�
;gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc*
is_constant(*
T0*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
�
<gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_2^gradients/Add*
_output_shapes
:*
swap_memory( *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
T0

�
Dgradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*
_output_shapes	
:�*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2
�
2gradients/encoder_1/rnn/while/Select_2_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_3_grad/Switch:16gradients/encoder_1/rnn/while/Select_2_grad/zeros_like*
T0* 
_output_shapes
:
��
�
4gradients/encoder_1/rnn/while/Select_2_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_2_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_3_grad/Switch:1*
T0* 
_output_shapes
:
��
�
4gradients/encoder_1/initial_state_3_tiled_grad/ShapeConst*
valueB"   �   *
_output_shapes
:*
dtype0
�
4gradients/encoder_1/initial_state_3_tiled_grad/stackPack)encoder_1/initial_state_3_tiled/multiples4gradients/encoder_1/initial_state_3_tiled_grad/Shape*
T0*

axis *
N*
_output_shapes

:
�
=gradients/encoder_1/initial_state_3_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_3_tiled_grad/stack*
T0*
_output_shapes
: 
�
>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub/yConst*
dtype0*
_output_shapes
: *
value	B :
�
<gradients/encoder_1/initial_state_3_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_3_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub/y*
_output_shapes
: *
T0
�
Dgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
Dgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
�
>gradients/encoder_1/initial_state_3_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_3_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/delta*
_output_shapes
:*

Tidx0
�
>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_3_tiled_grad/transpose/Range*
T0*
_output_shapes
:
�
8gradients/encoder_1/initial_state_3_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_3_tiled_grad/stack>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub_1*
Tperm0*
T0*
_output_shapes

:
�
<gradients/encoder_1/initial_state_3_tiled_grad/Reshape/shapeConst*
valueB:
���������*
_output_shapes
:*
dtype0
�
6gradients/encoder_1/initial_state_3_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_3_tiled_grad/transpose<gradients/encoder_1/initial_state_3_tiled_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
u
3gradients/encoder_1/initial_state_3_tiled_grad/SizeConst*
value	B :*
_output_shapes
: *
dtype0
|
:gradients/encoder_1/initial_state_3_tiled_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
|
:gradients/encoder_1/initial_state_3_tiled_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
�
4gradients/encoder_1/initial_state_3_tiled_grad/rangeRange:gradients/encoder_1/initial_state_3_tiled_grad/range/start3gradients/encoder_1/initial_state_3_tiled_grad/Size:gradients/encoder_1/initial_state_3_tiled_grad/range/delta*
_output_shapes
:*

Tidx0
�
8gradients/encoder_1/initial_state_3_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_5_grad/Exit6gradients/encoder_1/initial_state_3_tiled_grad/Reshape*
T0*
Tshape0*J
_output_shapes8
6:4������������������������������������
�
2gradients/encoder_1/initial_state_3_tiled_grad/SumSum8gradients/encoder_1/initial_state_3_tiled_grad/Reshape_14gradients/encoder_1/initial_state_3_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	�
�
<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5
�
?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
@gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_5^gradients/Add*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*
_output_shapes
:*
swap_memory( *
T0
�
Hgradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*(
_output_shapes
:����������
�
6gradients/encoder_1/rnn/while/Select_4_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop*
T0*(
_output_shapes
:����������
�
8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*

stack_name *
_output_shapes
:
�
;gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
T0
�
<gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_4^gradients/Add*
T0
*
_output_shapes
:*
swap_memory( *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4
�
Dgradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
T0*
is_constant(
�
;gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
_output_shapes	
:�
�
2gradients/encoder_1/rnn/while/Select_4_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_5_grad/Switch:16gradients/encoder_1/rnn/while/Select_4_grad/zeros_like* 
_output_shapes
:
��*
T0
�
4gradients/encoder_1/rnn/while/Select_4_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_4_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_5_grad/Switch:1* 
_output_shapes
:
��*
T0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
valueB"�   �   *
dtype0*
_output_shapes
:
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
valueB"�   �   *
_output_shapes
:*
dtype0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_accStack*
	elem_type0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
_output_shapes
:*

stack_name 
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnter:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1^gradients/Add*
_output_shapes
:*
swap_memory( *M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
T0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1* 
_output_shapes
:
��
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mulMul4gradients/encoder_1/rnn/while/Select_4_grad/Select_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop*
T0* 
_output_shapes
:
��
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
T0*
is_constant(
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2^gradients/Add*
_output_shapes
:*
swap_memory( *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2* 
_output_shapes
:
��
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop4gradients/encoder_1/rnn/while/Select_4_grad/Select_1* 
_output_shapes
:
��*
T0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*
T0*
Tshape0* 
_output_shapes
:
��
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape* 
_output_shapes
:
��*
T0
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1* 
_output_shapes
:
��*
T0
�
gradients/AddN_1AddN4gradients/encoder_1/rnn/while/Select_3_grad/Select_1Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGrad*
N*
T0* 
_output_shapes
:
��*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_3_grad/Select_1
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
valueB"�   �   *
dtype0*
_output_shapes
:
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
valueB"�   �   *
dtype0*
_output_shapes
:
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/SumSumgradients/AddN_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape* 
_output_shapes
:
��*
Tshape0*
T0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1* 
_output_shapes
:
��*
Tshape0*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"�   �   
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1Shapeencoder_1/rnn/while/Identity_4*
out_type0*
_output_shapes
:*
T0
�
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStack*
	elem_type0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*

stack_name 
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1
�
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1^gradients/Add*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*
swap_memory( *
T0
�
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
:*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1
�
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shapeegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*2
_output_shapes 
:���������:���������
�
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mulMulPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop*
T0* 
_output_shapes
:
��
�
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumSumJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape*
Tshape0* 
_output_shapes
:
��*
T0
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
T0
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnter;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid^gradients/Add*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
_output_shapes
:*
swap_memory( *
T0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1MulUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape* 
_output_shapes
:
��*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*(
_output_shapes
:����������*
Tshape0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
valueB"�   �   *
_output_shapes
:*
dtype0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"�   �   
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_accStack*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*

stack_name *
_output_shapes
:
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc*
is_constant(*
T0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnter8encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh^gradients/Add*
_output_shapes
:*
swap_memory( *K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mulMulRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop* 
_output_shapes
:
��*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape*
Tshape0* 
_output_shapes
:
��*
T0
�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
T0*
is_constant(
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1^gradients/Add*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
_output_shapes
:*
swap_memory( 
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1*
T0* 
_output_shapes
:
��
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1*
Tshape0* 
_output_shapes
:
��*
T0
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape* 
_output_shapes
:
��*
T0
�
gradients/AddN_2AddN2gradients/encoder_1/rnn/while/Select_3_grad/SelectPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1*
N*
T0* 
_output_shapes
:
��*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_3_grad/Select
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape*
T0* 
_output_shapes
:
��
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1* 
_output_shapes
:
��*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"�   �   
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
valueB *
_output_shapes
: *
dtype0
�
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumSumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape* 
_output_shapes
:
��*
Tshape0*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1SumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
;gradients/encoder_1/rnn/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_2* 
_output_shapes
:
��*
T0
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_accStack*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
_output_shapes
:*

stack_name 
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim
�
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPush	StackPushXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnterCencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPopStackPopagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
: *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim
�
Ogradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concatConcatV2Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop* 
_output_shapes
:
��*
T0*

Tidx0*
N
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
_output_shapes	
:�*
T0*
data_formatNHWC
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read* 
_output_shapes
:
��*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMulMatMulOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(* 
_output_shapes
:
��*
transpose_a( *
T0
�
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_accStack*
	elem_type0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
_output_shapes
:*

stack_name 
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat
�
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
T0
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat
�
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1MatMulegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
transpose_b( * 
_output_shapes
:
��*
transpose_a(*
T0
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes	
:�*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIteration*
_output_shapes
	:�: *
N*
T0
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*"
_output_shapes
:�:�*
T0
�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/AddAddYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch:1Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:�
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:�*
T0
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/RankConst^gradients/Sub*
dtype0*
_output_shapes
: *
value	B :
�
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
T0*
is_constant(
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPush	StackPush`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnterIencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis^gradients/Add*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*
swap_memory( *
T0
�
igradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopStackPopigradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnter^gradients/Sub*
	elem_type0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
_output_shapes
: 
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/modFloorMod`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
�
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"�   �   
�
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Shape_1Shapeencoder_1/rnn/while/Identity_5*
out_type0*
_output_shapes
:*
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_accStack*
	elem_type0*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:*

stack_name 
�
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPush	StackPushcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2^gradients/Add*
_output_shapes
:*
swap_memory( *L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
T0
�
lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopStackPoplgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnter^gradients/Sub*
	elem_type0*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2* 
_output_shapes
:
��
�
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeNShapeNcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop*
T0*
out_type0*
N* 
_output_shapes
::
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffsetConcatOffsetWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/modZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
�
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/SliceSliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffsetZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN*
Index0*
T0* 
_output_shapes
:
��
�
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice_1SliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMulbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffset:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:����������
�
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
��*    *
dtype0* 
_output_shapes
:
��
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1Enter_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc* 
_output_shapes
:
��*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2Mergeagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIteration*
N*
T0*"
_output_shapes
:
��: 
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/SwitchSwitchagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
��:
��
�
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/AddAddbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIterationNextIteration]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
��*
T0
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3Exit`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
��*
T0
�
gradients/AddN_3AddN4gradients/encoder_1/rnn/while/Select_2_grad/Select_1Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice*
N*
T0* 
_output_shapes
:
��*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_2_grad/Select_1
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"�   �   
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
valueB"�   �   *
dtype0*
_output_shapes
:
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnter:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1^gradients/Add*
_output_shapes
:*
swap_memory( *M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mulMulgradients/AddN_3Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop* 
_output_shapes
:
��*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape* 
_output_shapes
:
��*
Tshape0*
T0
�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2^gradients/Add*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
_output_shapes
:*
swap_memory( *
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2* 
_output_shapes
:
��
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopgradients/AddN_3*
T0* 
_output_shapes
:
��
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*
T0* 
_output_shapes
:
��*
Tshape0
�
gradients/AddN_4AddN2gradients/encoder_1/rnn/while/Select_4_grad/Select[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice_1*
N*
T0* 
_output_shapes
:
��*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_4_grad/Select
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape*
T0* 
_output_shapes
:
��
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1*
T0* 
_output_shapes
:
��
�
;gradients/encoder_1/rnn/while/Switch_5_grad_1/NextIterationNextIterationgradients/AddN_4* 
_output_shapes
:
��*
T0
�
gradients/AddN_5AddN4gradients/encoder_1/rnn/while/Select_1_grad/Select_1Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGrad*
T0*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_1_grad/Select_1*
N* 
_output_shapes
:
��
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
valueB"�   �   *
dtype0*
_output_shapes
:
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"�   �   
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/SumSumgradients/AddN_5^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape*
Tshape0* 
_output_shapes
:
��*
T0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_5`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*
Tshape0* 
_output_shapes
:
��*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"�   �   
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1Shapeencoder_1/rnn/while/Identity_2*
T0*
_output_shapes
:*
out_type0
�
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1
�
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1
�
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
:*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1
�
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shapeegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*2
_output_shapes 
:���������:���������
�
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mulMulPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop*
T0* 
_output_shapes
:
��
�
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumSumJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape*
Tshape0* 
_output_shapes
:
��*
T0
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc*
T0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnter;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid^gradients/Add*
_output_shapes
:*
swap_memory( *N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
T0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
T0*
is_constant(
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid* 
_output_shapes
:
��
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1MulUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape* 
_output_shapes
:
��*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*(
_output_shapes
:����������*
Tshape0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"�   �   
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
valueB"�   �   *
_output_shapes
:*
dtype0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnter8encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh^gradients/Add*
T0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
_output_shapes
:*
swap_memory( 
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mulMulRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop* 
_output_shapes
:
��*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_accStack*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
_output_shapes
:*

stack_name 
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc*
is_constant(*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1* 
_output_shapes
:
��*
T0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1* 
_output_shapes
:
��*
Tshape0*
T0
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape* 
_output_shapes
:
��*
T0
�
gradients/AddN_6AddN2gradients/encoder_1/rnn/while/Select_1_grad/SelectPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1*
N*
T0* 
_output_shapes
:
��*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_1_grad/Select
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape* 
_output_shapes
:
��*
T0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1* 
_output_shapes
:
��*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"�   �   
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
_output_shapes
: *
dtype0*
valueB 
�
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumSumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape* 
_output_shapes
:
��*
Tshape0*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1SumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
;gradients/encoder_1/rnn/while/Switch_2_grad_1/NextIterationNextIterationgradients/AddN_6*
T0* 
_output_shapes
:
��
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_accStack*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*

stack_name *
_output_shapes
:
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPush	StackPushXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnterCencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
T0
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPopStackPopagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
: *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim
�
Ogradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concatConcatV2Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop* 
_output_shapes
:
��*
T0*

Tidx0*
N
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
��*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMulMatMulOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(* 
_output_shapes
:
��*
transpose_a( *
T0
�
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
�
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
�
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
�
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1MatMulegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
transpose_b( * 
_output_shapes
:
��*
transpose_a(*
T0
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB�*    *
_output_shapes	
:�*
dtype0
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc*
is_constant( *
_output_shapes	
:�*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIteration*
_output_shapes
	:�: *
N*
T0
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
:�:�
�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/AddAddYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch:1Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:�
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:�
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/RankConst^gradients/Sub*
_output_shapes
: *
dtype0*
value	B :
�
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_accStack*
	elem_type0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*

stack_name *
_output_shapes
:
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
T0*
is_constant(
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPush	StackPush`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnterIencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis^gradients/Add*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*
swap_memory( *
T0
�
igradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopStackPopigradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
: *\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/modFloorMod`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
�
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"�   �   
�
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Shape_1Shapeencoder_1/rnn/while/Identity_3*
_output_shapes
:*
out_type0*
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_accStack*
	elem_type0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*

stack_name *
_output_shapes
:
�
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
�
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPush	StackPushcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter%encoder_1/rnn/while/TensorArrayReadV3^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *.
_class$
" loc:@encoder_1/rnn/TensorArray_1
�
lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopStackPoplgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnter^gradients/Sub*
	elem_type0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1* 
_output_shapes
:
��
�
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeNShapeNcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop*
out_type0* 
_output_shapes
::*
T0*
N
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffsetConcatOffsetWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/modZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
�
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/SliceSliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffsetZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN*
Index0*
T0* 
_output_shapes
:
��
�
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Slice_1SliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMulbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffset:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:����������
�
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
��*    *
dtype0* 
_output_shapes
:
��
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1Enter_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc*
is_constant( * 
_output_shapes
:
��*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2Mergeagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIteration*"
_output_shapes
:
��: *
N*
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/SwitchSwitchagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
��:
��
�
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/AddAddbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIterationNextIteration]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
��*
T0
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3Exit`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
��
�
\gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterencoder_1/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
�
^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterHencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
�
Vgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3\gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*
_output_shapes

::*
source	gradients*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
�
Rgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1W^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *.
_class$
" loc:@encoder_1/rnn/TensorArray_1
�
^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity
�
agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnterRefEnter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
bgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPush	StackPushagradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnterencoder_1/rnn/while/Identity^gradients/Add*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
:*
swap_memory( *
T0
�
jgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop/RefEnterRefEnter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopStackPopjgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
: *Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity
�
Xgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Vgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/SliceRgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: *.
_class$
" loc:@encoder_1/rnn/TensorArray_1
�
gradients/AddN_7AddN2gradients/encoder_1/rnn/while/Select_2_grad/Select[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Slice_1* 
_output_shapes
:
��*
N*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_2_grad/Select*
T0
�
Bgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterBgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
parallel_iterations *
T0*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
�
Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Jgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
�
Cgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*
T0*
_output_shapes
: : 
�
@gradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddEgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Xgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
�
Jgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration@gradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
_output_shapes
: *
T0
�
Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitCgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
_output_shapes
: *
T0
�
;gradients/encoder_1/rnn/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_7* 
_output_shapes
:
��*
T0
�
ygradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3encoder_1/rnn/TensorArray_1Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
source	gradients*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes

::
�
ugradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3z^gradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: 
�
kgradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3ygradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3&encoder_1/rnn/TensorArrayUnstack/rangeugradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*%
_output_shapes
:���*
element_shape:*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
�
4gradients/encoder_1/transpose_grad/InvertPermutationInvertPermutationencoder_1/transpose/perm*
_output_shapes
:*
T0
�
,gradients/encoder_1/transpose_grad/transpose	Transposekgradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV34gradients/encoder_1/transpose_grad/InvertPermutation*
Tperm0*%
_output_shapes
:���*
T0
�
+gradients/encoder/conv1d/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"�      �   �   
�
-gradients/encoder/conv1d/Squeeze_grad/ReshapeReshape,gradients/encoder_1/transpose_grad/transpose+gradients/encoder/conv1d/Squeeze_grad/Shape*
T0*)
_output_shapes
:���*
Tshape0
�
*gradients/encoder/conv1d/Conv2D_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"�      �      
�
8gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/encoder/conv1d/Conv2D_grad/Shapeencoder/conv1d/ExpandDims_1-gradients/encoder/conv1d/Squeeze_grad/Reshape*
use_cudnn_on_gpu(*(
_output_shapes
:��*
data_formatNHWC*
strides
*
T0*
paddingVALID
�
,gradients/encoder/conv1d/Conv2D_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"         �   
�
9gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/conv1d/ExpandDims,gradients/encoder/conv1d/Conv2D_grad/Shape_1-gradients/encoder/conv1d/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
0gradients/encoder/conv1d/ExpandDims_1_grad/ShapeConst*!
valueB"      �   *
dtype0*
_output_shapes
:
�
2gradients/encoder/conv1d/ExpandDims_1_grad/ReshapeReshape9gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropFilter0gradients/encoder/conv1d/ExpandDims_1_grad/Shape*#
_output_shapes
:�*
Tshape0*
T0
�
global_norm/L2LossL2Loss2gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*
_output_shapes
: *
T0
�
global_norm/L2Loss_1L2Loss2gradients/encoder_1/initial_state_0_tiled_grad/Sum*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
: *
T0
�
global_norm/L2Loss_2L2Loss2gradients/encoder_1/initial_state_1_tiled_grad/Sum*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
_output_shapes
: *
T0
�
global_norm/L2Loss_3L2Loss2gradients/encoder_1/initial_state_2_tiled_grad/Sum*
_output_shapes
: *E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum*
T0
�
global_norm/L2Loss_4L2Loss2gradients/encoder_1/initial_state_3_tiled_grad/Sum*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
_output_shapes
: *
T0
�
global_norm/L2Loss_5L2Lossagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
global_norm/L2Loss_6L2LossXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: 
�
global_norm/L2Loss_7L2Lossagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*
_output_shapes
: *t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3
�
global_norm/L2Loss_8L2LossXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: 
�
global_norm/L2Loss_9L2Loss:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
T0*
_output_shapes
: *M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1
�
global_norm/L2Loss_10L2Loss6gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
_output_shapes
: *I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
T0
�
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3global_norm/L2Loss_4global_norm/L2Loss_5global_norm/L2Loss_6global_norm/L2Loss_7global_norm/L2Loss_8global_norm/L2Loss_9global_norm/L2Loss_10*
N*
T0*
_output_shapes
:*

axis 
[
global_norm/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
X
global_norm/Const_1Const*
valueB
 *   @*
_output_shapes
: *
dtype0
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
T0*
_output_shapes
: 
Q
global_norm/global_normSqrtglobal_norm/mul*
T0*
_output_shapes
: 
b
clip_by_global_norm/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 

clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xglobal_norm/global_norm*
_output_shapes
: *
T0
^
clip_by_global_norm/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
d
clip_by_global_norm/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
�
clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
_output_shapes
: *
T0
�
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
�
clip_by_global_norm/mul_1Mul2gradients/encoder/conv1d/ExpandDims_1_grad/Reshapeclip_by_global_norm/mul*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*#
_output_shapes
:�*
T0
�
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*
T0*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*#
_output_shapes
:�
�
clip_by_global_norm/mul_2Mul2gradients/encoder_1/initial_state_0_tiled_grad/Sumclip_by_global_norm/mul*
_output_shapes
:	�*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
T0
�
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
:	�
�
clip_by_global_norm/mul_3Mul2gradients/encoder_1/initial_state_1_tiled_grad/Sumclip_by_global_norm/mul*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
_output_shapes
:	�*
T0
�
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
_output_shapes
:	�*
T0
�
clip_by_global_norm/mul_4Mul2gradients/encoder_1/initial_state_2_tiled_grad/Sumclip_by_global_norm/mul*
T0*
_output_shapes
:	�*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum
�
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
T0*
_output_shapes
:	�*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum
�
clip_by_global_norm/mul_5Mul2gradients/encoder_1/initial_state_3_tiled_grad/Sumclip_by_global_norm/mul*
T0*
_output_shapes
:	�*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum
�
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
_output_shapes
:	�
�
clip_by_global_norm/mul_6Mulagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0* 
_output_shapes
:
��*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3
�
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*
T0*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
��
�
clip_by_global_norm/mul_7MulXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
_output_shapes	
:�*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
*clip_by_global_norm/clip_by_global_norm/_6Identityclip_by_global_norm/mul_7*
T0*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:�
�
clip_by_global_norm/mul_8Mulagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul* 
_output_shapes
:
��*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
*clip_by_global_norm/clip_by_global_norm/_7Identityclip_by_global_norm/mul_8*
T0*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
��
�
clip_by_global_norm/mul_9MulXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:�*
T0
�
*clip_by_global_norm/clip_by_global_norm/_8Identityclip_by_global_norm/mul_9*
T0*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:�
�
clip_by_global_norm/mul_10Mul:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1clip_by_global_norm/mul*
T0*
_output_shapes
:	�*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1
�
*clip_by_global_norm/clip_by_global_norm/_9Identityclip_by_global_norm/mul_10*
_output_shapes
:	�*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
T0
�
clip_by_global_norm/mul_11Mul6gradients/output_projection/xw_plus_b_grad/BiasAddGradclip_by_global_norm/mul*I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
_output_shapes
:*
T0
�
+clip_by_global_norm/clip_by_global_norm/_10Identityclip_by_global_norm/mul_11*
T0*
_output_shapes
:*I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad
p
grad_norms/grad_norms/tagsConst*&
valueB Bgrad_norms/grad_norms*
dtype0*
_output_shapes
: 
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
�
beta1_power
VariableV2*
	container *
dtype0*
_class
loc:@input_embed*
_output_shapes
: *
shape: *
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@input_embed*
validate_shape(*
_output_shapes
: 
j
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
_class
loc:@input_embed*
T0
~
beta2_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *w�?*
_class
loc:@input_embed
�
beta2_power
VariableV2*
	container *
dtype0*
_class
loc:@input_embed*
shared_name *
_output_shapes
: *
shape: 
�
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
T0*
_output_shapes
: *
_class
loc:@input_embed
d
zerosConst*"
valueB�*    *
dtype0*#
_output_shapes
:�
�
input_embed/Adam
VariableV2*
shared_name *
shape:�*#
_output_shapes
:�*
_class
loc:@input_embed*
dtype0*
	container 
�
input_embed/Adam/AssignAssigninput_embed/Adamzeros*#
_output_shapes
:�*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
�
input_embed/Adam/readIdentityinput_embed/Adam*#
_output_shapes
:�*
_class
loc:@input_embed*
T0
f
zeros_1Const*"
valueB�*    *
dtype0*#
_output_shapes
:�
�
input_embed/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@input_embed*
shared_name *#
_output_shapes
:�*
shape:�
�
input_embed/Adam_1/AssignAssigninput_embed/Adam_1zeros_1*
use_locking(*
validate_shape(*
T0*#
_output_shapes
:�*
_class
loc:@input_embed
�
input_embed/Adam_1/readIdentityinput_embed/Adam_1*#
_output_shapes
:�*
_class
loc:@input_embed*
T0
^
zeros_2Const*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
encoder/initial_state_0/Adam
VariableV2*
shared_name *
shape:	�*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_0*
dtype0*
	container 
�
#encoder/initial_state_0/Adam/AssignAssignencoder/initial_state_0/Adamzeros_2**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
!encoder/initial_state_0/Adam/readIdentityencoder/initial_state_0/Adam*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_0*
T0
^
zeros_3Const*
valueB	�*    *
_output_shapes
:	�*
dtype0
�
encoder/initial_state_0/Adam_1
VariableV2**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	�*
shape:	�*
dtype0*
shared_name *
	container 
�
%encoder/initial_state_0/Adam_1/AssignAssignencoder/initial_state_0/Adam_1zeros_3**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
#encoder/initial_state_0/Adam_1/readIdentityencoder/initial_state_0/Adam_1*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_0
^
zeros_4Const*
valueB	�*    *
_output_shapes
:	�*
dtype0
�
encoder/initial_state_1/Adam
VariableV2**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	�*
shape:	�*
dtype0*
shared_name *
	container 
�
#encoder/initial_state_1/Adam/AssignAssignencoder/initial_state_1/Adamzeros_4**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
!encoder/initial_state_1/Adam/readIdentityencoder/initial_state_1/Adam*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_1*
T0
^
zeros_5Const*
_output_shapes
:	�*
dtype0*
valueB	�*    
�
encoder/initial_state_1/Adam_1
VariableV2*
shape:	�*
_output_shapes
:	�*
shared_name **
_class 
loc:@encoder/initial_state_1*
dtype0*
	container 
�
%encoder/initial_state_1/Adam_1/AssignAssignencoder/initial_state_1/Adam_1zeros_5*
_output_shapes
:	�*
validate_shape(**
_class 
loc:@encoder/initial_state_1*
T0*
use_locking(
�
#encoder/initial_state_1/Adam_1/readIdentityencoder/initial_state_1/Adam_1*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_1
^
zeros_6Const*
_output_shapes
:	�*
dtype0*
valueB	�*    
�
encoder/initial_state_2/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:	�*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_2
�
#encoder/initial_state_2/Adam/AssignAssignencoder/initial_state_2/Adamzeros_6*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_2*
validate_shape(*
_output_shapes
:	�
�
!encoder/initial_state_2/Adam/readIdentityencoder/initial_state_2/Adam*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_2*
T0
^
zeros_7Const*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
encoder/initial_state_2/Adam_1
VariableV2**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	�*
shape:	�*
dtype0*
shared_name *
	container 
�
%encoder/initial_state_2/Adam_1/AssignAssignencoder/initial_state_2/Adam_1zeros_7**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
#encoder/initial_state_2/Adam_1/readIdentityencoder/initial_state_2/Adam_1*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_2
^
zeros_8Const*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
encoder/initial_state_3/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:	�*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_3
�
#encoder/initial_state_3/Adam/AssignAssignencoder/initial_state_3/Adamzeros_8**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
!encoder/initial_state_3/Adam/readIdentityencoder/initial_state_3/Adam**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	�*
T0
^
zeros_9Const*
_output_shapes
:	�*
dtype0*
valueB	�*    
�
encoder/initial_state_3/Adam_1
VariableV2*
shared_name *
shape:	�*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_3*
dtype0*
	container 
�
%encoder/initial_state_3/Adam_1/AssignAssignencoder/initial_state_3/Adam_1zeros_9*
_output_shapes
:	�*
validate_shape(**
_class 
loc:@encoder/initial_state_3*
T0*
use_locking(
�
#encoder/initial_state_3/Adam_1/readIdentityencoder/initial_state_3/Adam_1*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_3*
T0
a
zeros_10Const*
dtype0* 
_output_shapes
:
��*
valueB
��*    
�
8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:
��* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
�
?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/AssignAssign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamzeros_10* 
_output_shapes
:
��*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0*
use_locking(
�
=encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/readIdentity8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0
a
zeros_11Const*
valueB
��*    *
dtype0* 
_output_shapes
:
��
�
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1
VariableV2*
	container *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
��*
shape:
��*
shared_name 
�
Aencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/AssignAssign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1zeros_11*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
�
?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/readIdentity:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
��
W
zeros_12Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam
VariableV2*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adamzeros_12*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
�
<encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/readIdentity7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:�
W
zeros_13Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1
VariableV2*
shape:�*
_output_shapes	
:�*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
dtype0*
	container 
�
@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1zeros_13*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
validate_shape(*
_output_shapes	
:�
�
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/readIdentity9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*
T0*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
a
zeros_14Const*
valueB
��*    * 
_output_shapes
:
��*
dtype0
�
8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam
VariableV2*
	container *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
shared_name * 
_output_shapes
:
��*
shape:
��
�
?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/AssignAssign8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adamzeros_14*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
��*
T0*
validate_shape(*
use_locking(
�
=encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/readIdentity8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
��
a
zeros_15Const* 
_output_shapes
:
��*
dtype0*
valueB
��*    
�
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1
VariableV2*
shared_name *
shape:
��* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
	container 
�
Aencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/AssignAssign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1zeros_15*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
�
?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/readIdentity:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
��
W
zeros_16Const*
valueB�*    *
_output_shapes	
:�*
dtype0
�
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam
VariableV2*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:�*
shape:�*
dtype0*
shared_name *
	container 
�
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adamzeros_16*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
�
<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/readIdentity7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:�*
T0
W
zeros_17Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1
VariableV2*
shape:�*
_output_shapes	
:�*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
dtype0*
	container 
�
@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1zeros_17*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
validate_shape(*
_output_shapes	
:�
�
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/readIdentity9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:�*
T0
_
zeros_18Const*
valueB	�*    *
_output_shapes
:	�*
dtype0
�
output_projection/W/Adam
VariableV2*
shape:	�*
_output_shapes
:	�*
shared_name *&
_class
loc:@output_projection/W*
dtype0*
	container 
�
output_projection/W/Adam/AssignAssignoutput_projection/W/Adamzeros_18*
_output_shapes
:	�*
validate_shape(*&
_class
loc:@output_projection/W*
T0*
use_locking(
�
output_projection/W/Adam/readIdentityoutput_projection/W/Adam*&
_class
loc:@output_projection/W*
_output_shapes
:	�*
T0
_
zeros_19Const*
valueB	�*    *
_output_shapes
:	�*
dtype0
�
output_projection/W/Adam_1
VariableV2*
shared_name *
shape:	�*
_output_shapes
:	�*&
_class
loc:@output_projection/W*
dtype0*
	container 
�
!output_projection/W/Adam_1/AssignAssignoutput_projection/W/Adam_1zeros_19*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�*&
_class
loc:@output_projection/W
�
output_projection/W/Adam_1/readIdentityoutput_projection/W/Adam_1*
_output_shapes
:	�*&
_class
loc:@output_projection/W*
T0
U
zeros_20Const*
dtype0*
_output_shapes
:*
valueB*    
�
output_projection/b/Adam
VariableV2*
shared_name *&
_class
loc:@output_projection/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
output_projection/b/Adam/AssignAssignoutput_projection/b/Adamzeros_20*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
�
output_projection/b/Adam/readIdentityoutput_projection/b/Adam*
T0*
_output_shapes
:*&
_class
loc:@output_projection/b
U
zeros_21Const*
valueB*    *
dtype0*
_output_shapes
:
�
output_projection/b/Adam_1
VariableV2*
	container *
dtype0*&
_class
loc:@output_projection/b*
shared_name *
_output_shapes
:*
shape:
�
!output_projection/b/Adam_1/AssignAssignoutput_projection/b/Adam_1zeros_21*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*&
_class
loc:@output_projection/b
�
output_projection/b/Adam_1/readIdentityoutput_projection/b/Adam_1*
T0*&
_class
loc:@output_projection/b*
_output_shapes
:
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *w�?
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
!Adam/update_input_embed/ApplyAdam	ApplyAdaminput_embedinput_embed/Adaminput_embed/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*#
_output_shapes
:�*
_class
loc:@input_embed*
T0*
use_locking( 
�
-Adam/update_encoder/initial_state_0/ApplyAdam	ApplyAdamencoder/initial_state_0encoder/initial_state_0/Adamencoder/initial_state_0/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_locking( *
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_0
�
-Adam/update_encoder/initial_state_1/ApplyAdam	ApplyAdamencoder/initial_state_1encoder/initial_state_1/Adamencoder/initial_state_1/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
use_locking( *
T0**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	�
�
-Adam/update_encoder/initial_state_2/ApplyAdam	ApplyAdamencoder/initial_state_2encoder/initial_state_2/Adamencoder/initial_state_2/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
use_locking( *
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_2
�
-Adam/update_encoder/initial_state_3/ApplyAdam	ApplyAdamencoder/initial_state_3encoder/initial_state_3/Adamencoder/initial_state_3/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_4*
use_locking( *
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_3
�
IAdam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdam	ApplyAdam3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_5*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
��*
T0*
use_locking( 
�
HAdam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdam	ApplyAdam2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_6*
use_locking( *
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:�
�
IAdam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdam	ApplyAdam3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_7*
use_locking( *
T0* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
�
HAdam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam	ApplyAdam2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_8*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
use_locking( 
�
)Adam/update_output_projection/W/ApplyAdam	ApplyAdamoutput_projection/Woutput_projection/W/Adamoutput_projection/W/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_9*
_output_shapes
:	�*&
_class
loc:@output_projection/W*
T0*
use_locking( 
�
)Adam/update_output_projection/b/ApplyAdam	ApplyAdamoutput_projection/boutput_projection/b/Adamoutput_projection/b/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_10*
use_locking( *
T0*&
_class
loc:@output_projection/b*
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam*
_class
loc:@input_embed*
_output_shapes
: *
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@input_embed
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam*
_class
loc:@input_embed*
_output_shapes
: *
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@input_embed*
validate_shape(*
_output_shapes
: 
�
Adam/updateNoOp"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam^Adam/Assign^Adam/Assign_1
w

Adam/valueConst^Adam/update*
value	B :*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
x
Adam	AssignAddVariable
Adam/value*
_class
loc:@Variable*
_output_shapes
: *
T0*
use_locking( 
�
Const_1Const*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                *
dtype0	*
_output_shapes	
:�
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
:�
E
EqualEqualArgMaxConst_1*
T0	*
_output_shapes	
:�
L

LogicalAnd
LogicalAndaccuracy/EqualEqual*
_output_shapes	
:�
Z
count_nonzero/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

n
count_nonzero/NotEqualNotEqual
LogicalAndcount_nonzero/NotEqual/y*
T0
*
_output_shapes	
:�
j
count_nonzero/ToInt64Castcount_nonzero/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

]
count_nonzero/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
count_nonzero/SumSumcount_nonzero/ToInt64count_nonzero/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
Y
Equal_1Equaloutput_projection/ArgMaxConst_1*
_output_shapes	
:�*
T0	
\
count_nonzero_1/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
o
count_nonzero_1/NotEqualNotEqualEqual_1count_nonzero_1/NotEqual/y*
T0
*
_output_shapes	
:�
n
count_nonzero_1/ToInt64Castcount_nonzero_1/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
_
count_nonzero_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
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
ArgMax_1ArgMaxcond/Merge_1ArgMax_1/dimension*

Tidx0*
T0*
_output_shapes	
:�
I
Equal_2EqualArgMax_1Const_1*
_output_shapes	
:�*
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
:�
n
count_nonzero_2/ToInt64Castcount_nonzero_2/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

_
count_nonzero_2/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
count_nonzero_2/SumSumcount_nonzero_2/ToInt64count_nonzero_2/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
K
	Greater/yConst*
_output_shapes
: *
dtype0	*
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

�
cond_1/truediv/Cast/SwitchSwitchcount_nonzero/Sumcond_1/pred_id*$
_class
loc:@count_nonzero/Sum*
_output_shapes
: : *
T0	
i
cond_1/truediv/CastCastcond_1/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
�
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
cond_1/MergeMergecond_1/Constcond_1/truediv*
T0*
N*
_output_shapes
: : 
\
precision_0/tagsConst*
valueB Bprecision_0*
_output_shapes
: *
dtype0
]
precision_0ScalarSummaryprecision_0/tagscond_1/Merge*
T0*
_output_shapes
: 
M
Greater_1/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
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

�
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
�
cond_2/truediv/Cast_1/SwitchSwitchcount_nonzero_2/Sumcond_2/pred_id*
_output_shapes
: : *&
_class
loc:@count_nonzero_2/Sum*
T0	
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
valueB 2        *
_output_shapes
: *
dtype0
_
cond_2/MergeMergecond_2/Constcond_2/truediv*
_output_shapes
: : *
T0*
N
V
recall_0/tagsConst*
dtype0*
_output_shapes
: *
valueB Brecall_0
W
recall_0ScalarSummaryrecall_0/tagscond_2/Merge*
_output_shapes
: *
T0
�
Const_2Const*
dtype0	*
_output_shapes	
:�*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
T
ArgMax_2/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
f
ArgMax_2ArgMaxcond/Merge_1ArgMax_2/dimension*
_output_shapes	
:�*
T0*

Tidx0
I
Equal_3EqualArgMax_2Const_2*
T0	*
_output_shapes	
:�
P
LogicalAnd_1
LogicalAndaccuracy/EqualEqual_3*
_output_shapes	
:�
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
:�
n
count_nonzero_3/ToInt64Castcount_nonzero_3/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
_
count_nonzero_3/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
count_nonzero_3/SumSumcount_nonzero_3/ToInt64count_nonzero_3/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
Y
Equal_4Equaloutput_projection/ArgMaxConst_2*
_output_shapes	
:�*
T0	
\
count_nonzero_4/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

o
count_nonzero_4/NotEqualNotEqualEqual_4count_nonzero_4/NotEqual/y*
T0
*
_output_shapes	
:�
n
count_nonzero_4/ToInt64Castcount_nonzero_4/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

_
count_nonzero_4/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
count_nonzero_4/SumSumcount_nonzero_4/ToInt64count_nonzero_4/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
T
ArgMax_3/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
f
ArgMax_3ArgMaxcond/Merge_1ArgMax_3/dimension*

Tidx0*
T0*
_output_shapes	
:�
I
Equal_5EqualArgMax_3Const_2*
T0	*
_output_shapes	
:�
\
count_nonzero_5/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
o
count_nonzero_5/NotEqualNotEqualEqual_5count_nonzero_5/NotEqual/y*
_output_shapes	
:�*
T0

n
count_nonzero_5/ToInt64Castcount_nonzero_5/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
_
count_nonzero_5/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
count_nonzero_5/SumSumcount_nonzero_5/ToInt64count_nonzero_5/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
M
Greater_2/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
W
	Greater_2Greatercount_nonzero_4/SumGreater_2/y*
_output_shapes
: *
T0	
P
cond_3/SwitchSwitch	Greater_2	Greater_2*
_output_shapes
: : *
T0

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
cond_3/pred_idIdentity	Greater_2*
_output_shapes
: *
T0

�
cond_3/truediv/Cast/SwitchSwitchcount_nonzero_3/Sumcond_3/pred_id*
_output_shapes
: : *&
_class
loc:@count_nonzero_3/Sum*
T0	
i
cond_3/truediv/CastCastcond_3/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
�
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
cond_3/ConstConst^cond_3/switch_f*
valueB 2        *
_output_shapes
: *
dtype0
_
cond_3/MergeMergecond_3/Constcond_3/truediv*
_output_shapes
: : *
N*
T0
\
precision_1/tagsConst*
valueB Bprecision_1*
dtype0*
_output_shapes
: 
]
precision_1ScalarSummaryprecision_1/tagscond_3/Merge*
T0*
_output_shapes
: 
M
Greater_3/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
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
cond_4/switch_tIdentitycond_4/Switch:1*
_output_shapes
: *
T0

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

�
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
�
cond_4/truediv/Cast_1/SwitchSwitchcount_nonzero_5/Sumcond_4/pred_id*
T0	*&
_class
loc:@count_nonzero_5/Sum*
_output_shapes
: : 
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
N*
T0*
_output_shapes
: : 
V
recall_1/tagsConst*
_output_shapes
: *
dtype0*
valueB Brecall_1
W
recall_1ScalarSummaryrecall_1/tagscond_4/Merge*
T0*
_output_shapes
: 
�
Const_3Const*
dtype0	*
_output_shapes	
:�*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
T
ArgMax_4/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
f
ArgMax_4ArgMaxcond/Merge_1ArgMax_4/dimension*

Tidx0*
T0*
_output_shapes	
:�
I
Equal_6EqualArgMax_4Const_3*
_output_shapes	
:�*
T0	
P
LogicalAnd_2
LogicalAndaccuracy/EqualEqual_6*
_output_shapes	
:�
\
count_nonzero_6/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

t
count_nonzero_6/NotEqualNotEqualLogicalAnd_2count_nonzero_6/NotEqual/y*
_output_shapes	
:�*
T0

n
count_nonzero_6/ToInt64Castcount_nonzero_6/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

_
count_nonzero_6/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
count_nonzero_6/SumSumcount_nonzero_6/ToInt64count_nonzero_6/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
Y
Equal_7Equaloutput_projection/ArgMaxConst_3*
_output_shapes	
:�*
T0	
\
count_nonzero_7/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
o
count_nonzero_7/NotEqualNotEqualEqual_7count_nonzero_7/NotEqual/y*
_output_shapes	
:�*
T0

n
count_nonzero_7/ToInt64Castcount_nonzero_7/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
_
count_nonzero_7/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
count_nonzero_7/SumSumcount_nonzero_7/ToInt64count_nonzero_7/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
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
:�
I
Equal_8EqualArgMax_5Const_3*
T0	*
_output_shapes	
:�
\
count_nonzero_8/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
o
count_nonzero_8/NotEqualNotEqualEqual_8count_nonzero_8/NotEqual/y*
_output_shapes	
:�*
T0

n
count_nonzero_8/ToInt64Castcount_nonzero_8/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

_
count_nonzero_8/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
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
	Greater_4Greatercount_nonzero_7/SumGreater_4/y*
_output_shapes
: *
T0	
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
cond_5/switch_fIdentitycond_5/Switch*
T0
*
_output_shapes
: 
F
cond_5/pred_idIdentity	Greater_4*
_output_shapes
: *
T0

�
cond_5/truediv/Cast/SwitchSwitchcount_nonzero_6/Sumcond_5/pred_id*
_output_shapes
: : *&
_class
loc:@count_nonzero_6/Sum*
T0	
i
cond_5/truediv/CastCastcond_5/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
�
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
cond_5/ConstConst^cond_5/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
_
cond_5/MergeMergecond_5/Constcond_5/truediv*
_output_shapes
: : *
N*
T0
\
precision_2/tagsConst*
valueB Bprecision_2*
dtype0*
_output_shapes
: 
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
cond_6/switch_fIdentitycond_6/Switch*
_output_shapes
: *
T0

F
cond_6/pred_idIdentity	Greater_5*
T0
*
_output_shapes
: 
�
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
�
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
cond_6/ConstConst^cond_6/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
_
cond_6/MergeMergecond_6/Constcond_6/truediv*
_output_shapes
: : *
T0*
N
V
recall_2/tagsConst*
dtype0*
_output_shapes
: *
valueB Brecall_2
W
recall_2ScalarSummaryrecall_2/tagscond_6/Merge*
_output_shapes
: *
T0
�
Const_4Const*
dtype0	*
_output_shapes	
:�*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
T
ArgMax_6/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
f
ArgMax_6ArgMaxcond/Merge_1ArgMax_6/dimension*

Tidx0*
T0*
_output_shapes	
:�
I
Equal_9EqualArgMax_6Const_4*
_output_shapes	
:�*
T0	
P
LogicalAnd_3
LogicalAndaccuracy/EqualEqual_9*
_output_shapes	
:�
\
count_nonzero_9/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
t
count_nonzero_9/NotEqualNotEqualLogicalAnd_3count_nonzero_9/NotEqual/y*
_output_shapes	
:�*
T0

n
count_nonzero_9/ToInt64Castcount_nonzero_9/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
_
count_nonzero_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
count_nonzero_9/SumSumcount_nonzero_9/ToInt64count_nonzero_9/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
Z
Equal_10Equaloutput_projection/ArgMaxConst_4*
_output_shapes	
:�*
T0	
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
:�
p
count_nonzero_10/ToInt64Castcount_nonzero_10/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
`
count_nonzero_10/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
count_nonzero_10/SumSumcount_nonzero_10/ToInt64count_nonzero_10/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
T
ArgMax_7/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
f
ArgMax_7ArgMaxcond/Merge_1ArgMax_7/dimension*
_output_shapes	
:�*
T0*

Tidx0
J
Equal_11EqualArgMax_7Const_4*
T0	*
_output_shapes	
:�
]
count_nonzero_11/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
r
count_nonzero_11/NotEqualNotEqualEqual_11count_nonzero_11/NotEqual/y*
T0
*
_output_shapes	
:�
p
count_nonzero_11/ToInt64Castcount_nonzero_11/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

`
count_nonzero_11/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
count_nonzero_11/SumSumcount_nonzero_11/ToInt64count_nonzero_11/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
M
Greater_6/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
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
cond_7/switch_fIdentitycond_7/Switch*
T0
*
_output_shapes
: 
F
cond_7/pred_idIdentity	Greater_6*
_output_shapes
: *
T0

�
cond_7/truediv/Cast/SwitchSwitchcount_nonzero_9/Sumcond_7/pred_id*
T0	*
_output_shapes
: : *&
_class
loc:@count_nonzero_9/Sum
i
cond_7/truediv/CastCastcond_7/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
�
cond_7/truediv/Cast_1/SwitchSwitchcount_nonzero_10/Sumcond_7/pred_id*
T0	*
_output_shapes
: : *'
_class
loc:@count_nonzero_10/Sum
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
valueB Bprecision_3*
_output_shapes
: *
dtype0
]
precision_3ScalarSummaryprecision_3/tagscond_7/Merge*
_output_shapes
: *
T0
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
cond_8/SwitchSwitch	Greater_7	Greater_7*
T0
*
_output_shapes
: : 
M
cond_8/switch_tIdentitycond_8/Switch:1*
_output_shapes
: *
T0

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

�
cond_8/truediv/Cast/SwitchSwitchcount_nonzero_9/Sumcond_8/pred_id*
T0	*
_output_shapes
: : *&
_class
loc:@count_nonzero_9/Sum
i
cond_8/truediv/CastCastcond_8/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
�
cond_8/truediv/Cast_1/SwitchSwitchcount_nonzero_11/Sumcond_8/pred_id*
T0	*'
_class
loc:@count_nonzero_11/Sum*
_output_shapes
: : 
m
cond_8/truediv/Cast_1Castcond_8/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
f
cond_8/truedivRealDivcond_8/truediv/Castcond_8/truediv/Cast_1*
_output_shapes
: *
T0
g
cond_8/ConstConst^cond_8/switch_f*
valueB 2        *
dtype0*
_output_shapes
: 
_
cond_8/MergeMergecond_8/Constcond_8/truediv*
_output_shapes
: : *
N*
T0
V
recall_3/tagsConst*
_output_shapes
: *
dtype0*
valueB Brecall_3
W
recall_3ScalarSummaryrecall_3/tagscond_8/Merge*
_output_shapes
: *
T0
�
Const_5Const*
_output_shapes	
:�*
dtype0	*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
T
ArgMax_8/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
f
ArgMax_8ArgMaxcond/Merge_1ArgMax_8/dimension*
_output_shapes	
:�*
T0*

Tidx0
J
Equal_12EqualArgMax_8Const_5*
_output_shapes	
:�*
T0	
Q
LogicalAnd_4
LogicalAndaccuracy/EqualEqual_12*
_output_shapes	
:�
]
count_nonzero_12/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
v
count_nonzero_12/NotEqualNotEqualLogicalAnd_4count_nonzero_12/NotEqual/y*
T0
*
_output_shapes	
:�
p
count_nonzero_12/ToInt64Castcount_nonzero_12/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
`
count_nonzero_12/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
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
:�
]
count_nonzero_13/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
r
count_nonzero_13/NotEqualNotEqualEqual_13count_nonzero_13/NotEqual/y*
_output_shapes	
:�*
T0

p
count_nonzero_13/ToInt64Castcount_nonzero_13/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
`
count_nonzero_13/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
count_nonzero_13/SumSumcount_nonzero_13/ToInt64count_nonzero_13/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
T
ArgMax_9/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
f
ArgMax_9ArgMaxcond/Merge_1ArgMax_9/dimension*
_output_shapes	
:�*
T0*

Tidx0
J
Equal_14EqualArgMax_9Const_5*
T0	*
_output_shapes	
:�
]
count_nonzero_14/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

r
count_nonzero_14/NotEqualNotEqualEqual_14count_nonzero_14/NotEqual/y*
T0
*
_output_shapes	
:�
p
count_nonzero_14/ToInt64Castcount_nonzero_14/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

`
count_nonzero_14/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
count_nonzero_14/SumSumcount_nonzero_14/ToInt64count_nonzero_14/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
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
cond_9/SwitchSwitch	Greater_8	Greater_8*
T0
*
_output_shapes
: : 
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
cond_9/pred_idIdentity	Greater_8*
_output_shapes
: *
T0

�
cond_9/truediv/Cast/SwitchSwitchcount_nonzero_12/Sumcond_9/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_12/Sum*
T0	
i
cond_9/truediv/CastCastcond_9/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
�
cond_9/truediv/Cast_1/SwitchSwitchcount_nonzero_13/Sumcond_9/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_13/Sum*
T0	
m
cond_9/truediv/Cast_1Castcond_9/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
f
cond_9/truedivRealDivcond_9/truediv/Castcond_9/truediv/Cast_1*
T0*
_output_shapes
: 
g
cond_9/ConstConst^cond_9/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
_
cond_9/MergeMergecond_9/Constcond_9/truediv*
_output_shapes
: : *
N*
T0
\
precision_4/tagsConst*
valueB Bprecision_4*
dtype0*
_output_shapes
: 
]
precision_4ScalarSummaryprecision_4/tagscond_9/Merge*
T0*
_output_shapes
: 
M
Greater_9/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
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
cond_10/pred_idIdentity	Greater_9*
T0
*
_output_shapes
: 
�
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
�
cond_10/truediv/Cast_1/SwitchSwitchcount_nonzero_14/Sumcond_10/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_14/Sum*
T0	
o
cond_10/truediv/Cast_1Castcond_10/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
i
cond_10/truedivRealDivcond_10/truediv/Castcond_10/truediv/Cast_1*
T0*
_output_shapes
: 
i
cond_10/ConstConst^cond_10/switch_f*
valueB 2        *
dtype0*
_output_shapes
: 
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
recall_4ScalarSummaryrecall_4/tagscond_10/Merge*
T0*
_output_shapes
: 
�
Const_6Const*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                *
_output_shapes	
:�*
dtype0	
U
ArgMax_10/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
h
	ArgMax_10ArgMaxcond/Merge_1ArgMax_10/dimension*

Tidx0*
T0*
_output_shapes	
:�
K
Equal_15Equal	ArgMax_10Const_6*
T0	*
_output_shapes	
:�
Q
LogicalAnd_5
LogicalAndaccuracy/EqualEqual_15*
_output_shapes	
:�
]
count_nonzero_15/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
v
count_nonzero_15/NotEqualNotEqualLogicalAnd_5count_nonzero_15/NotEqual/y*
T0
*
_output_shapes	
:�
p
count_nonzero_15/ToInt64Castcount_nonzero_15/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

`
count_nonzero_15/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
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
:�
]
count_nonzero_16/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
r
count_nonzero_16/NotEqualNotEqualEqual_16count_nonzero_16/NotEqual/y*
_output_shapes	
:�*
T0

p
count_nonzero_16/ToInt64Castcount_nonzero_16/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
`
count_nonzero_16/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
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
	ArgMax_11ArgMaxcond/Merge_1ArgMax_11/dimension*

Tidx0*
T0*
_output_shapes	
:�
K
Equal_17Equal	ArgMax_11Const_6*
_output_shapes	
:�*
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
count_nonzero_17/NotEqualNotEqualEqual_17count_nonzero_17/NotEqual/y*
_output_shapes	
:�*
T0

p
count_nonzero_17/ToInt64Castcount_nonzero_17/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

`
count_nonzero_17/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
count_nonzero_17/SumSumcount_nonzero_17/ToInt64count_nonzero_17/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
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

�
cond_11/truediv/Cast/SwitchSwitchcount_nonzero_15/Sumcond_11/pred_id*
T0	*'
_class
loc:@count_nonzero_15/Sum*
_output_shapes
: : 
k
cond_11/truediv/CastCastcond_11/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
�
cond_11/truediv/Cast_1/SwitchSwitchcount_nonzero_16/Sumcond_11/pred_id*'
_class
loc:@count_nonzero_16/Sum*
_output_shapes
: : *
T0	
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
valueB 2        *
dtype0*
_output_shapes
: 
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
Greater_11/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
Z

Greater_11Greatercount_nonzero_17/SumGreater_11/y*
_output_shapes
: *
T0	
S
cond_12/SwitchSwitch
Greater_11
Greater_11*
T0
*
_output_shapes
: : 
O
cond_12/switch_tIdentitycond_12/Switch:1*
T0
*
_output_shapes
: 
M
cond_12/switch_fIdentitycond_12/Switch*
_output_shapes
: *
T0

H
cond_12/pred_idIdentity
Greater_11*
T0
*
_output_shapes
: 
�
cond_12/truediv/Cast/SwitchSwitchcount_nonzero_15/Sumcond_12/pred_id*'
_class
loc:@count_nonzero_15/Sum*
_output_shapes
: : *
T0	
k
cond_12/truediv/CastCastcond_12/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
�
cond_12/truediv/Cast_1/SwitchSwitchcount_nonzero_17/Sumcond_12/pred_id*
T0	*
_output_shapes
: : *'
_class
loc:@count_nonzero_17/Sum
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
cond_12/MergeMergecond_12/Constcond_12/truediv*
_output_shapes
: : *
T0*
N
V
recall_5/tagsConst*
valueB Brecall_5*
dtype0*
_output_shapes
: 
X
recall_5ScalarSummaryrecall_5/tagscond_12/Merge*
T0*
_output_shapes
: 
�
Const_7Const*
_output_shapes	
:�*
dtype0	*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
U
ArgMax_12/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
h
	ArgMax_12ArgMaxcond/Merge_1ArgMax_12/dimension*

Tidx0*
T0*
_output_shapes	
:�
K
Equal_18Equal	ArgMax_12Const_7*
T0	*
_output_shapes	
:�
Q
LogicalAnd_6
LogicalAndaccuracy/EqualEqual_18*
_output_shapes	
:�
]
count_nonzero_18/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
v
count_nonzero_18/NotEqualNotEqualLogicalAnd_6count_nonzero_18/NotEqual/y*
_output_shapes	
:�*
T0

p
count_nonzero_18/ToInt64Castcount_nonzero_18/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
`
count_nonzero_18/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
count_nonzero_18/SumSumcount_nonzero_18/ToInt64count_nonzero_18/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
Z
Equal_19Equaloutput_projection/ArgMaxConst_7*
_output_shapes	
:�*
T0	
]
count_nonzero_19/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
r
count_nonzero_19/NotEqualNotEqualEqual_19count_nonzero_19/NotEqual/y*
_output_shapes	
:�*
T0

p
count_nonzero_19/ToInt64Castcount_nonzero_19/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
`
count_nonzero_19/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
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
	ArgMax_13ArgMaxcond/Merge_1ArgMax_13/dimension*
_output_shapes	
:�*
T0*

Tidx0
K
Equal_20Equal	ArgMax_13Const_7*
_output_shapes	
:�*
T0	
]
count_nonzero_20/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
r
count_nonzero_20/NotEqualNotEqualEqual_20count_nonzero_20/NotEqual/y*
_output_shapes	
:�*
T0

p
count_nonzero_20/ToInt64Castcount_nonzero_20/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
`
count_nonzero_20/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
count_nonzero_20/SumSumcount_nonzero_20/ToInt64count_nonzero_20/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
N
Greater_12/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
Z

Greater_12Greatercount_nonzero_19/SumGreater_12/y*
T0	*
_output_shapes
: 
S
cond_13/SwitchSwitch
Greater_12
Greater_12*
_output_shapes
: : *
T0

O
cond_13/switch_tIdentitycond_13/Switch:1*
_output_shapes
: *
T0

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

�
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
�
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
cond_13/ConstConst^cond_13/switch_f*
valueB 2        *
_output_shapes
: *
dtype0
b
cond_13/MergeMergecond_13/Constcond_13/truediv*
N*
T0*
_output_shapes
: : 
\
precision_6/tagsConst*
dtype0*
_output_shapes
: *
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

Greater_13Greatercount_nonzero_20/SumGreater_13/y*
T0	*
_output_shapes
: 
S
cond_14/SwitchSwitch
Greater_13
Greater_13*
T0
*
_output_shapes
: : 
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
�
cond_14/truediv/Cast/SwitchSwitchcount_nonzero_18/Sumcond_14/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_18/Sum*
T0	
k
cond_14/truediv/CastCastcond_14/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
�
cond_14/truediv/Cast_1/SwitchSwitchcount_nonzero_20/Sumcond_14/pred_id*'
_class
loc:@count_nonzero_20/Sum*
_output_shapes
: : *
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
cond_14/ConstConst^cond_14/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
b
cond_14/MergeMergecond_14/Constcond_14/truediv*
T0*
N*
_output_shapes
: : 
V
recall_6/tagsConst*
dtype0*
_output_shapes
: *
valueB Brecall_6
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
lr/tagsConst*
dtype0*
_output_shapes
: *
value
B Blr
L
lrScalarSummarylr/tagslearning_rate*
_output_shapes
: *
T0
Z
accuracy_1/tagsConst*
dtype0*
_output_shapes
: *
valueB B
accuracy_1
`

accuracy_1ScalarSummaryaccuracy_1/tagsaccuracy/accuracy*
_output_shapes
: *
T0
�
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

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
�

save/SaveV2/tensor_namesConst*
_output_shapes
:$*
dtype0*�

value�	B�	$BVariableBbeta1_powerBbeta2_powerBencoder/initial_state_0Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1Binput_embedBinput_embed/AdamBinput_embed/Adam_1Boutput_projection/WBoutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/bBoutput_projection/b/AdamBoutput_projection/b/Adam_1
�
save/SaveV2/shape_and_slicesConst*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:$*
dtype0
�

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
save/RestoreV2/tensor_namesConst*
valueBBVariable*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@Variable
q
save/RestoreV2_1/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assignbeta1_powersave/RestoreV2_1*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@input_embed
q
save/RestoreV2_2/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
!save/RestoreV2_3/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assignencoder/initial_state_0save/RestoreV2_3*
_output_shapes
:	�*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(
�
save/RestoreV2_4/tensor_namesConst*
_output_shapes
:*
dtype0*1
value(B&Bencoder/initial_state_0/Adam
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_4Assignencoder/initial_state_0/Adamsave/RestoreV2_4*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_0
�
save/RestoreV2_5/tensor_namesConst*
dtype0*
_output_shapes
:*3
value*B(Bencoder/initial_state_0/Adam_1
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_5Assignencoder/initial_state_0/Adam_1save/RestoreV2_5*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_0*
validate_shape(*
_output_shapes
:	�
}
save/RestoreV2_6/tensor_namesConst*
_output_shapes
:*
dtype0*,
value#B!Bencoder/initial_state_1
j
!save/RestoreV2_6/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_6Assignencoder/initial_state_1save/RestoreV2_6*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	�
�
save/RestoreV2_7/tensor_namesConst*
_output_shapes
:*
dtype0*1
value(B&Bencoder/initial_state_1/Adam
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7Assignencoder/initial_state_1/Adamsave/RestoreV2_7*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	�
�
save/RestoreV2_8/tensor_namesConst*
_output_shapes
:*
dtype0*3
value*B(Bencoder/initial_state_1/Adam_1
j
!save/RestoreV2_8/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_8Assignencoder/initial_state_1/Adam_1save/RestoreV2_8*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	�
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
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_9Assignencoder/initial_state_2save/RestoreV2_9*
_output_shapes
:	�*
validate_shape(**
_class 
loc:@encoder/initial_state_2*
T0*
use_locking(
�
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
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_10Assignencoder/initial_state_2/Adamsave/RestoreV2_10**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_11/tensor_namesConst*3
value*B(Bencoder/initial_state_2/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_11/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_11Assignencoder/initial_state_2/Adam_1save/RestoreV2_11*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_2
~
save/RestoreV2_12/tensor_namesConst*
_output_shapes
:*
dtype0*,
value#B!Bencoder/initial_state_3
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_12Assignencoder/initial_state_3save/RestoreV2_12*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_3
�
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
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_13Assignencoder/initial_state_3/Adamsave/RestoreV2_13**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_14/tensor_namesConst*3
value*B(Bencoder/initial_state_3/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assignencoder/initial_state_3/Adam_1save/RestoreV2_14*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_3
�
save/RestoreV2_15/tensor_namesConst*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes
:*
dtype0
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_15Assign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasessave/RestoreV2_15*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:�*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_16/tensor_namesConst*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_16Assign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adamsave/RestoreV2_16*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:�*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_17/tensor_namesConst*N
valueEBCB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_17Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1save/RestoreV2_17*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
validate_shape(*
_output_shapes	
:�
�
save/RestoreV2_18/tensor_namesConst*
dtype0*
_output_shapes
:*H
value?B=B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
k
"save/RestoreV2_18/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_18Assign3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightssave/RestoreV2_18*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
�
save/RestoreV2_19/tensor_namesConst*
dtype0*
_output_shapes
:*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam
k
"save/RestoreV2_19/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_19Assign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamsave/RestoreV2_19* 
_output_shapes
:
��*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0*
use_locking(
�
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
�
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_20Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1save/RestoreV2_20*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
validate_shape(* 
_output_shapes
:
��
�
save/RestoreV2_21/tensor_namesConst*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_21Assign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasessave/RestoreV2_21*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
validate_shape(*
_output_shapes	
:�
�
save/RestoreV2_22/tensor_namesConst*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_22/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_22Assign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adamsave/RestoreV2_22*
_output_shapes	
:�*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
use_locking(
�
save/RestoreV2_23/tensor_namesConst*
dtype0*
_output_shapes
:*N
valueEBCB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1
k
"save/RestoreV2_23/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_23Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1save/RestoreV2_23*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
�
save/RestoreV2_24/tensor_namesConst*
dtype0*
_output_shapes
:*H
value?B=B3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
k
"save/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_24Assign3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightssave/RestoreV2_24* 
_output_shapes
:
��*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0*
use_locking(
�
save/RestoreV2_25/tensor_namesConst*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_25/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_25Assign8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adamsave/RestoreV2_25* 
_output_shapes
:
��*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0*
use_locking(
�
save/RestoreV2_26/tensor_namesConst*O
valueFBDB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_26Assign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1save/RestoreV2_26* 
_output_shapes
:
��*
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
"save/RestoreV2_27/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_27Assigninput_embedsave/RestoreV2_27*#
_output_shapes
:�*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
w
save/RestoreV2_28/tensor_namesConst*
dtype0*
_output_shapes
:*%
valueBBinput_embed/Adam
k
"save/RestoreV2_28/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_28Assigninput_embed/Adamsave/RestoreV2_28*
use_locking(*
validate_shape(*
T0*#
_output_shapes
:�*
_class
loc:@input_embed
y
save/RestoreV2_29/tensor_namesConst*'
valueBBinput_embed/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_29Assigninput_embed/Adam_1save/RestoreV2_29*
use_locking(*
T0*
_class
loc:@input_embed*
validate_shape(*#
_output_shapes
:�
z
save/RestoreV2_30/tensor_namesConst*(
valueBBoutput_projection/W*
_output_shapes
:*
dtype0
k
"save/RestoreV2_30/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_30Assignoutput_projection/Wsave/RestoreV2_30*&
_class
loc:@output_projection/W*
_output_shapes
:	�*
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
"save/RestoreV2_31/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_31Assignoutput_projection/W/Adamsave/RestoreV2_31*
_output_shapes
:	�*
validate_shape(*&
_class
loc:@output_projection/W*
T0*
use_locking(
�
save/RestoreV2_32/tensor_namesConst*/
value&B$Boutput_projection/W/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_32/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_32Assignoutput_projection/W/Adam_1save/RestoreV2_32*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�*&
_class
loc:@output_projection/W
z
save/RestoreV2_33/tensor_namesConst*(
valueBBoutput_projection/b*
_output_shapes
:*
dtype0
k
"save/RestoreV2_33/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_33Assignoutput_projection/bsave/RestoreV2_33*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

save/RestoreV2_34/tensor_namesConst*-
value$B"Boutput_projection/b/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_34/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_34Assignoutput_projection/b/Adamsave/RestoreV2_34*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(
�
save/RestoreV2_35/tensor_namesConst*
_output_shapes
:*
dtype0*/
value&B$Boutput_projection/b/Adam_1
k
"save/RestoreV2_35/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_35Assignoutput_projection/b/Adam_1save/RestoreV2_35*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedVariable*
_class
loc:@Variable*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedinput_embed*
_class
loc:@input_embed*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedencoder/initial_state_0*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_0
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedencoder/initial_state_1**
_class 
loc:@encoder/initial_state_1*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedencoder/initial_state_2*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_2
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedencoder/initial_state_3**
_class 
loc:@encoder/initial_state_3*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitialized2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
dtype0*
_output_shapes
: *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedoutput_projection/W*&
_class
loc:@output_projection/W*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedoutput_projection/b*&
_class
loc:@output_projection/b*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedbeta1_power*
_class
loc:@input_embed*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedbeta2_power*
_class
loc:@input_embed*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedinput_embed/Adam*
dtype0*
_output_shapes
: *
_class
loc:@input_embed
�
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedinput_embed/Adam_1*
dtype0*
_output_shapes
: *
_class
loc:@input_embed
�
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedencoder/initial_state_0/Adam**
_class 
loc:@encoder/initial_state_0*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitializedencoder/initial_state_0/Adam_1*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_0
�
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitializedencoder/initial_state_1/Adam*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_1
�
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitializedencoder/initial_state_1/Adam_1**
_class 
loc:@encoder/initial_state_1*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitializedencoder/initial_state_2/Adam*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_2
�
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitializedencoder/initial_state_2/Adam_1*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_2
�
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitializedencoder/initial_state_3/Adam**
_class 
loc:@encoder/initial_state_3*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitializedencoder/initial_state_3/Adam_1**
_class 
loc:@encoder/initial_state_3*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_24IsVariableInitialized8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_25IsVariableInitialized:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1*
_output_shapes
: *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
�
7report_uninitialized_variables/IsVariableInitialized_26IsVariableInitialized7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*
dtype0*
_output_shapes
: *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
�
7report_uninitialized_variables/IsVariableInitialized_27IsVariableInitialized9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
�
7report_uninitialized_variables/IsVariableInitialized_28IsVariableInitialized8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam*
_output_shapes
: *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
�
7report_uninitialized_variables/IsVariableInitialized_29IsVariableInitialized:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_30IsVariableInitialized7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_31IsVariableInitialized9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
�
7report_uninitialized_variables/IsVariableInitialized_32IsVariableInitializedoutput_projection/W/Adam*&
_class
loc:@output_projection/W*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_33IsVariableInitializedoutput_projection/W/Adam_1*
dtype0*
_output_shapes
: *&
_class
loc:@output_projection/W
�
7report_uninitialized_variables/IsVariableInitialized_34IsVariableInitializedoutput_projection/b/Adam*&
_class
loc:@output_projection/b*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_35IsVariableInitializedoutput_projection/b/Adam_1*&
_class
loc:@output_projection/b*
_output_shapes
: *
dtype0
�
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_217report_uninitialized_variables/IsVariableInitialized_227report_uninitialized_variables/IsVariableInitialized_237report_uninitialized_variables/IsVariableInitialized_247report_uninitialized_variables/IsVariableInitialized_257report_uninitialized_variables/IsVariableInitialized_267report_uninitialized_variables/IsVariableInitialized_277report_uninitialized_variables/IsVariableInitialized_287report_uninitialized_variables/IsVariableInitialized_297report_uninitialized_variables/IsVariableInitialized_307report_uninitialized_variables/IsVariableInitialized_317report_uninitialized_variables/IsVariableInitialized_327report_uninitialized_variables/IsVariableInitialized_337report_uninitialized_variables/IsVariableInitialized_347report_uninitialized_variables/IsVariableInitialized_35*

axis *
_output_shapes
:$*
T0
*
N$
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:$
�

$report_uninitialized_variables/ConstConst*
dtype0*
_output_shapes
:$*�

value�	B�	$BVariableBinput_embedBencoder/initial_state_0Bencoder/initial_state_1Bencoder/initial_state_2Bencoder/initial_state_3B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesBoutput_projection/WBoutput_projection/bBbeta1_powerBbeta2_powerBinput_embed/AdamBinput_embed/Adam_1Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1B7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1Boutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/b/AdamBoutput_projection/b/Adam_1
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:$*
_output_shapes
:*
dtype0
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
T0*
Index0*
new_axis_mask *
_output_shapes
:*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
end_mask 
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:$
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
: 
�
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
_output_shapes
:*
N*

axis *
T0
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
_output_shapes
:$*
T0
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
valueB:
���������*
_output_shapes
:*
dtype0
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
T0
*
_output_shapes
:$*
Tshape0
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*#
_output_shapes
:���������*
validate_indices(*
Tparams0*
Tindices0	
�
initNoOp^Variable/Assign^input_embed/Assign^encoder/initial_state_0/Assign^encoder/initial_state_1/Assign^encoder/initial_state_2/Assign^encoder/initial_state_3/Assign;^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Assign:^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Assign;^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Assign:^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Assign^output_projection/W/Assign^output_projection/b/Assign^beta1_power/Assign^beta2_power/Assign^input_embed/Adam/Assign^input_embed/Adam_1/Assign$^encoder/initial_state_0/Adam/Assign&^encoder/initial_state_0/Adam_1/Assign$^encoder/initial_state_1/Adam/Assign&^encoder/initial_state_1/Adam_1/Assign$^encoder/initial_state_2/Adam/Assign&^encoder/initial_state_2/Adam_1/Assign$^encoder/initial_state_3/Adam/Assign&^encoder/initial_state_3/Adam_1/Assign@^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/AssignB^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/Assign?^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/AssignA^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/Assign@^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/AssignB^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/Assign?^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/AssignA^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/Assign ^output_projection/W/Adam/Assign"^output_projection/W/Adam_1/Assign ^output_projection/b/Adam/Assign"^output_projection/b/Adam_1/Assign

init_1NoOp

init_all_tablesNoOp
-

group_depsNoOp^init_1^init_all_tables"�����     4�y�	o!L�C�AJ��
�L�L
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
2	��
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
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�"
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
	summarizeint�
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
p
	AssignAdd
ref"T�

value"T

output_ref"T�"
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
�
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
�
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
�
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
�
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
�
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
ref"dtype�
is_initialized
"
dtypetype�
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
�
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
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
�
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
�
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
2	�
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PaddingFIFOQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint���������"
	containerstring "
shared_namestring �
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
�
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
�
QueueDequeueManyV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint���������
~
QueueDequeueV2

handle

components2component_types"!
component_types
list(type)(0"

timeout_msint���������
v
QueueEnqueueV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint���������
#
QueueSizeV2

handle
size
�
RandomShuffleQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint���������"
min_after_dequeueint "
seedint "
seed2int "
	containerstring "
shared_namestring �
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
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
�
RefEnter
data"T�
output"T�"	
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
handle�"
	elem_typetype"

stack_namestring �
?
StackPop
handle�
elem"	elem_type"
	elem_typetype
V
	StackPush
handle�	
elem"T
output"T"	
Ttype"
swap_memorybool( 
�
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
�
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
sourcestring�
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
�
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("
tensor_array_namestring �
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
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �

Where	
input
	
index	
&
	ZerosLike
x"T
y"T"	
Ttype*1.0.12v1.0.0-65-g4763edf-dirty��
f
PlaceholderPlaceholder*0
_output_shapes
:������������������*
shape: *
dtype0
[
Placeholder_1Placeholder*
shape: *
dtype0*#
_output_shapes
:���������
X

early_stopPlaceholder*
shape: *
dtype0*#
_output_shapes
:���������
�
random_queue_testRandomShuffleQueueV2*#
shapes
:	�::*
min_after_dequeue�*
capacity�
*

seed{*
	container *
seed2�*
shared_name *
_output_shapes
: *
component_types
2
�
random_queue_test_enqueueQueueEnqueueV2random_queue_testPlaceholderPlaceholder_1
early_stop*
Tcomponents
2*

timeout_ms���������
�
random_queue_test_DequeueQueueDequeueV2random_queue_test*

timeout_ms���������*+
_output_shapes
:	�::*
component_types
2
U
batch_and_pad/ConstConst*
dtype0
*
_output_shapes
: *
value	B
 Z
�
 batch_and_pad/padding_fifo_queuePaddingFIFOQueueV2*#
shapes
:	�::*
capacity�
*
	container *
shared_name *
_output_shapes
: *
component_types
2
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
batch_and_pad/cond/switch_fIdentitybatch_and_pad/cond/Switch*
T0
*
_output_shapes
: 
\
batch_and_pad/cond/pred_idIdentitybatch_and_pad/Const*
T0
*
_output_shapes
: 
�
4batch_and_pad/cond/padding_fifo_queue_enqueue/SwitchSwitch batch_and_pad/padding_fifo_queuebatch_and_pad/cond/pred_id*
T0*3
_class)
'%loc:@batch_and_pad/padding_fifo_queue*
_output_shapes
: : 
�
6batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_1Switchrandom_queue_test_Dequeuebatch_and_pad/cond/pred_id*,
_class"
 loc:@random_queue_test_Dequeue**
_output_shapes
:	�:	�*
T0
�
6batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_2Switchrandom_queue_test_Dequeue:1batch_and_pad/cond/pred_id*
T0*,
_class"
 loc:@random_queue_test_Dequeue* 
_output_shapes
::
�
6batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_3Switchrandom_queue_test_Dequeue:2batch_and_pad/cond/pred_id*
T0* 
_output_shapes
::*,
_class"
 loc:@random_queue_test_Dequeue
�
-batch_and_pad/cond/padding_fifo_queue_enqueueQueueEnqueueV26batch_and_pad/cond/padding_fifo_queue_enqueue/Switch:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_1:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_2:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_3:1*
Tcomponents
2*

timeout_ms���������
�
%batch_and_pad/cond/control_dependencyIdentitybatch_and_pad/cond/switch_t.^batch_and_pad/cond/padding_fifo_queue_enqueue*
_output_shapes
: *.
_class$
" loc:@batch_and_pad/cond/switch_t*
T0

=
batch_and_pad/cond/NoOpNoOp^batch_and_pad/cond/switch_f
�
'batch_and_pad/cond/control_dependency_1Identitybatch_and_pad/cond/switch_f^batch_and_pad/cond/NoOp*
_output_shapes
: *.
_class$
" loc:@batch_and_pad/cond/switch_f*
T0

�
batch_and_pad/cond/MergeMerge'batch_and_pad/cond/control_dependency_1%batch_and_pad/cond/control_dependency*
_output_shapes
: : *
T0
*
N
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
batch_and_pad/mulMulbatch_and_pad/Castbatch_and_pad/mul/y*
T0*
_output_shapes
: 
�
(batch_and_pad/fraction_of_1384_full/tagsConst*
_output_shapes
: *
dtype0*4
value+B) B#batch_and_pad/fraction_of_1384_full
�
#batch_and_pad/fraction_of_1384_fullScalarSummary(batch_and_pad/fraction_of_1384_full/tagsbatch_and_pad/mul*
_output_shapes
: *
T0
R
batch_and_pad/nConst*
value
B :�*
dtype0*
_output_shapes
: 
�
batch_and_padQueueDequeueManyV2 batch_and_pad/padding_fifo_queuebatch_and_pad/n*

timeout_ms���������*:
_output_shapes(
&:��:	�:	�*
component_types
2
h
Placeholder_2Placeholder*
shape: *
dtype0*0
_output_shapes
:������������������
[
Placeholder_3Placeholder*
shape: *
dtype0*#
_output_shapes
:���������
Z
early_stop_1Placeholder*#
_output_shapes
:���������*
shape: *
dtype0
�
random_queue_trainRandomShuffleQueueV2*#
shapes
:	�::*
_output_shapes
: *
component_types
2*
seed2{*
	container *
capacity�
*
min_after_dequeue�*
shared_name *

seed{
�
random_queue_train_enqueueQueueEnqueueV2random_queue_trainPlaceholder_2Placeholder_3early_stop_1*
Tcomponents
2*

timeout_ms���������
�
random_queue_train_DequeueQueueDequeueV2random_queue_train*+
_output_shapes
:	�::*
component_types
2*

timeout_ms���������
W
batch_and_pad_1/ConstConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
"batch_and_pad_1/padding_fifo_queuePaddingFIFOQueueV2*#
shapes
:	�::*
	container *
_output_shapes
: *
component_types
2*
capacity�
*
shared_name 
v
batch_and_pad_1/cond/SwitchSwitchbatch_and_pad_1/Constbatch_and_pad_1/Const*
_output_shapes
: : *
T0

i
batch_and_pad_1/cond/switch_tIdentitybatch_and_pad_1/cond/Switch:1*
T0
*
_output_shapes
: 
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

�
6batch_and_pad_1/cond/padding_fifo_queue_enqueue/SwitchSwitch"batch_and_pad_1/padding_fifo_queuebatch_and_pad_1/cond/pred_id*
T0*
_output_shapes
: : *5
_class+
)'loc:@batch_and_pad_1/padding_fifo_queue
�
8batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_1Switchrandom_queue_train_Dequeuebatch_and_pad_1/cond/pred_id*
T0**
_output_shapes
:	�:	�*-
_class#
!loc:@random_queue_train_Dequeue
�
8batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_2Switchrandom_queue_train_Dequeue:1batch_and_pad_1/cond/pred_id*
T0* 
_output_shapes
::*-
_class#
!loc:@random_queue_train_Dequeue
�
8batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_3Switchrandom_queue_train_Dequeue:2batch_and_pad_1/cond/pred_id*
T0* 
_output_shapes
::*-
_class#
!loc:@random_queue_train_Dequeue
�
/batch_and_pad_1/cond/padding_fifo_queue_enqueueQueueEnqueueV28batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch:1:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_1:1:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_2:1:batch_and_pad_1/cond/padding_fifo_queue_enqueue/Switch_3:1*
Tcomponents
2*

timeout_ms���������
�
'batch_and_pad_1/cond/control_dependencyIdentitybatch_and_pad_1/cond/switch_t0^batch_and_pad_1/cond/padding_fifo_queue_enqueue*0
_class&
$"loc:@batch_and_pad_1/cond/switch_t*
_output_shapes
: *
T0

A
batch_and_pad_1/cond/NoOpNoOp^batch_and_pad_1/cond/switch_f
�
)batch_and_pad_1/cond/control_dependency_1Identitybatch_and_pad_1/cond/switch_f^batch_and_pad_1/cond/NoOp*0
_class&
$"loc:@batch_and_pad_1/cond/switch_f*
_output_shapes
: *
T0

�
batch_and_pad_1/cond/MergeMerge)batch_and_pad_1/cond/control_dependency_1'batch_and_pad_1/cond/control_dependency*
_output_shapes
: : *
N*
T0

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
batch_and_pad_1/mulMulbatch_and_pad_1/Castbatch_and_pad_1/mul/y*
T0*
_output_shapes
: 
�
*batch_and_pad_1/fraction_of_1384_full/tagsConst*
dtype0*
_output_shapes
: *6
value-B+ B%batch_and_pad_1/fraction_of_1384_full
�
%batch_and_pad_1/fraction_of_1384_fullScalarSummary*batch_and_pad_1/fraction_of_1384_full/tagsbatch_and_pad_1/mul*
_output_shapes
: *
T0
T
batch_and_pad_1/nConst*
value
B :�*
dtype0*
_output_shapes
: 
�
batch_and_pad_1QueueDequeueManyV2"batch_and_pad_1/padding_fifo_queuebatch_and_pad_1/n*:
_output_shapes(
&:��:	�:	�*
component_types
2*

timeout_ms���������
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

�
cond/Switch_1Switchbatch_and_pad_1cond/pred_id*"
_class
loc:@batch_and_pad_1*4
_output_shapes"
 :��:��*
T0
�
cond/Switch_2Switchbatch_and_pad_1:1cond/pred_id*"
_class
loc:@batch_and_pad_1**
_output_shapes
:	�:	�*
T0
�
cond/Switch_3Switchbatch_and_pad_1:2cond/pred_id*"
_class
loc:@batch_and_pad_1**
_output_shapes
:	�:	�*
T0
�
cond/Switch_4Switchbatch_and_padcond/pred_id* 
_class
loc:@batch_and_pad*4
_output_shapes"
 :��:��*
T0
�
cond/Switch_5Switchbatch_and_pad:1cond/pred_id**
_output_shapes
:	�:	�* 
_class
loc:@batch_and_pad*
T0
�
cond/Switch_6Switchbatch_and_pad:2cond/pred_id*
T0* 
_class
loc:@batch_and_pad**
_output_shapes
:	�:	�
m

cond/MergeMergecond/Switch_4cond/Switch_1:1*&
_output_shapes
:��: *
N*
T0
j
cond/Merge_1Mergecond/Switch_5cond/Switch_2:1*
N*
T0*!
_output_shapes
:	�: 
j
cond/Merge_2Mergecond/Switch_6cond/Switch_3:1*!
_output_shapes
:	�: *
T0*
N
j
cond/Merge_3Mergecond/Switch_6cond/Switch_3:1*
T0*
N*!
_output_shapes
:	�: 
`
Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
c
ReshapeReshapecond/Merge_2Reshape/shape*
_output_shapes	
:�*
Tshape0*
T0
X
Variable/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
l
Variable
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
Variable/AssignAssignVariableVariable/initial_value*
_class
loc:@Variable*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
a
Variable/readIdentityVariable*
_class
loc:@Variable*
_output_shapes
: *
T0
�
,input_embed/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
_class
loc:@input_embed*!
valueB"      �   
�
*input_embed/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
_class
loc:@input_embed*
valueB
 *
ף�
�
*input_embed/Initializer/random_uniform/maxConst*
_class
loc:@input_embed*
valueB
 *
ף=*
dtype0*
_output_shapes
: 
�
4input_embed/Initializer/random_uniform/RandomUniformRandomUniform,input_embed/Initializer/random_uniform/shape*#
_output_shapes
:�*
_class
loc:@input_embed*
dtype0*

seed{*
T0*
seed2W
�
*input_embed/Initializer/random_uniform/subSub*input_embed/Initializer/random_uniform/max*input_embed/Initializer/random_uniform/min*
T0*
_output_shapes
: *
_class
loc:@input_embed
�
*input_embed/Initializer/random_uniform/mulMul4input_embed/Initializer/random_uniform/RandomUniform*input_embed/Initializer/random_uniform/sub*#
_output_shapes
:�*
_class
loc:@input_embed*
T0
�
&input_embed/Initializer/random_uniformAdd*input_embed/Initializer/random_uniform/mul*input_embed/Initializer/random_uniform/min*
_class
loc:@input_embed*#
_output_shapes
:�*
T0
�
input_embed
VariableV2*
_class
loc:@input_embed*#
_output_shapes
:�*
shape:�*
dtype0*
shared_name *
	container 
�
input_embed/AssignAssigninput_embed&input_embed/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@input_embed*
validate_shape(*#
_output_shapes
:�
w
input_embed/readIdentityinput_embed*#
_output_shapes
:�*
_class
loc:@input_embed*
T0
_
encoder/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
�
encoder/conv1d/ExpandDims
ExpandDims
cond/Mergeencoder/conv1d/ExpandDims/dim*

Tdim0*
T0*(
_output_shapes
:��
a
encoder/conv1d/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
encoder/conv1d/ExpandDims_1
ExpandDimsinput_embed/readencoder/conv1d/ExpandDims_1/dim*

Tdim0*'
_output_shapes
:�*
T0
�
encoder/conv1d/Conv2DConv2Dencoder/conv1d/ExpandDimsencoder/conv1d/ExpandDims_1*)
_output_shapes
:���*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
T0

encoder/conv1d/SqueezeSqueezeencoder/conv1d/Conv2D*
squeeze_dims
*%
_output_shapes
:���*
T0
Z
ShapeConst*
dtype0*
_output_shapes
:*!
valueB"�   �      
]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask*
Index0*
T0*
end_mask *
_output_shapes
: *

begin_mask *
ellipsis_mask 
�
)encoder/initial_state_0/Initializer/ConstConst*
_output_shapes
:	�*
dtype0**
_class 
loc:@encoder/initial_state_0*
valueB	�*    
�
encoder/initial_state_0
VariableV2*
shape:	�*
_output_shapes
:	�*
shared_name **
_class 
loc:@encoder/initial_state_0*
dtype0*
	container 
�
encoder/initial_state_0/AssignAssignencoder/initial_state_0)encoder/initial_state_0/Initializer/Const*
_output_shapes
:	�*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(
�
encoder/initial_state_0/readIdentityencoder/initial_state_0*
T0**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	�
m
+encoder_1/initial_state_0_tiled/multiples/1Const*
value	B :*
_output_shapes
: *
dtype0
�
)encoder_1/initial_state_0_tiled/multiplesPackstrided_slice+encoder_1/initial_state_0_tiled/multiples/1*
T0*

axis *
N*
_output_shapes
:
�
encoder_1/initial_state_0_tiledTileencoder/initial_state_0/read)encoder_1/initial_state_0_tiled/multiples*(
_output_shapes
:����������*
T0*

Tmultiples0
�
)encoder/initial_state_1/Initializer/ConstConst**
_class 
loc:@encoder/initial_state_1*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
encoder/initial_state_1
VariableV2**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	�*
shape:	�*
dtype0*
shared_name *
	container 
�
encoder/initial_state_1/AssignAssignencoder/initial_state_1)encoder/initial_state_1/Initializer/Const*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	�
�
encoder/initial_state_1/readIdentityencoder/initial_state_1*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_1*
T0
m
+encoder_1/initial_state_1_tiled/multiples/1Const*
dtype0*
_output_shapes
: *
value	B :
�
)encoder_1/initial_state_1_tiled/multiplesPackstrided_slice+encoder_1/initial_state_1_tiled/multiples/1*
T0*

axis *
N*
_output_shapes
:
�
encoder_1/initial_state_1_tiledTileencoder/initial_state_1/read)encoder_1/initial_state_1_tiled/multiples*

Tmultiples0*
T0*(
_output_shapes
:����������
�
)encoder/initial_state_2/Initializer/ConstConst**
_class 
loc:@encoder/initial_state_2*
valueB	�*    *
_output_shapes
:	�*
dtype0
�
encoder/initial_state_2
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_2*
shared_name *
_output_shapes
:	�*
shape:	�
�
encoder/initial_state_2/AssignAssignencoder/initial_state_2)encoder/initial_state_2/Initializer/Const*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_2*
validate_shape(*
_output_shapes
:	�
�
encoder/initial_state_2/readIdentityencoder/initial_state_2**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	�*
T0
m
+encoder_1/initial_state_2_tiled/multiples/1Const*
dtype0*
_output_shapes
: *
value	B :
�
)encoder_1/initial_state_2_tiled/multiplesPackstrided_slice+encoder_1/initial_state_2_tiled/multiples/1*
T0*

axis *
N*
_output_shapes
:
�
encoder_1/initial_state_2_tiledTileencoder/initial_state_2/read)encoder_1/initial_state_2_tiled/multiples*

Tmultiples0*
T0*(
_output_shapes
:����������
�
)encoder/initial_state_3/Initializer/ConstConst*
dtype0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_3*
valueB	�*    
�
encoder/initial_state_3
VariableV2*
	container *
shared_name *
dtype0*
shape:	�*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_3
�
encoder/initial_state_3/AssignAssignencoder/initial_state_3)encoder/initial_state_3/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_3
�
encoder/initial_state_3/readIdentityencoder/initial_state_3*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_3*
T0
m
+encoder_1/initial_state_3_tiled/multiples/1Const*
dtype0*
_output_shapes
: *
value	B :
�
)encoder_1/initial_state_3_tiled/multiplesPackstrided_slice+encoder_1/initial_state_3_tiled/multiples/1*
N*
T0*
_output_shapes
:*

axis 
�
encoder_1/initial_state_3_tiledTileencoder/initial_state_3/read)encoder_1/initial_state_3_tiled/multiples*(
_output_shapes
:����������*
T0*

Tmultiples0
m
encoder_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
�
encoder_1/transpose	Transposeencoder/conv1d/Squeezeencoder_1/transpose/perm*
Tperm0*%
_output_shapes
:���*
T0
T
encoder_1/sequence_lengthIdentityReshape*
T0*
_output_shapes	
:�
h
encoder_1/rnn/ShapeConst*
dtype0*
_output_shapes
:*!
valueB"�   �   �   
k
!encoder_1/rnn/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
m
#encoder_1/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
m
#encoder_1/rnn/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
encoder_1/rnn/strided_sliceStridedSliceencoder_1/rnn/Shape!encoder_1/rnn/strided_slice/stack#encoder_1/rnn/strided_slice/stack_1#encoder_1/rnn/strided_slice/stack_2*
end_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask*
_output_shapes
: *
new_axis_mask *
Index0*
T0
m
#encoder_1/rnn/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%encoder_1/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%encoder_1/rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
encoder_1/rnn/strided_slice_1StridedSliceencoder_1/rnn/Shape#encoder_1/rnn/strided_slice_1/stack%encoder_1/rnn/strided_slice_1/stack_1%encoder_1/rnn/strided_slice_1/stack_2*
T0*
Index0*
new_axis_mask *
_output_shapes
: *
shrink_axis_mask*
ellipsis_mask *

begin_mask *
end_mask 
`
encoder_1/rnn/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
r
encoder_1/rnn/stackPackencoder_1/rnn/strided_slice*
_output_shapes
:*
N*

axis *
T0
m
encoder_1/rnn/EqualEqualencoder_1/rnn/Shape_1encoder_1/rnn/stack*
_output_shapes
:*
T0
]
encoder_1/rnn/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
w
encoder_1/rnn/AllAllencoder_1/rnn/Equalencoder_1/rnn/Const*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
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
�
"encoder_1/rnn/Assert/Assert/data_0Const*J
valueAB? B9Expected shape for Tensor encoder_1/sequence_length:0 is *
_output_shapes
: *
dtype0
s
"encoder_1/rnn/Assert/Assert/data_2Const*!
valueB B but saw shape: *
_output_shapes
: *
dtype0
�
encoder_1/rnn/Assert/AssertAssertencoder_1/rnn/All"encoder_1/rnn/Assert/Assert/data_0encoder_1/rnn/stack"encoder_1/rnn/Assert/Assert/data_2encoder_1/rnn/Shape_1*
T
2*
	summarize
�
encoder_1/rnn/CheckSeqLenIdentityencoder_1/sequence_length^encoder_1/rnn/Assert/Assert*
_output_shapes	
:�*
T0
j
encoder_1/rnn/Shape_2Const*
dtype0*
_output_shapes
:*!
valueB"�   �   �   
m
#encoder_1/rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
o
%encoder_1/rnn/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
o
%encoder_1/rnn/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
encoder_1/rnn/strided_slice_2StridedSliceencoder_1/rnn/Shape_2#encoder_1/rnn/strided_slice_2/stack%encoder_1/rnn/strided_slice_2/stack_1%encoder_1/rnn/strided_slice_2/stack_2*
new_axis_mask *
shrink_axis_mask*
T0*
Index0*
end_mask *
_output_shapes
: *
ellipsis_mask *

begin_mask 
m
#encoder_1/rnn/strided_slice_3/stackConst*
valueB:*
_output_shapes
:*
dtype0
o
%encoder_1/rnn/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%encoder_1/rnn/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
encoder_1/rnn/strided_slice_3StridedSliceencoder_1/rnn/Shape_2#encoder_1/rnn/strided_slice_3/stack%encoder_1/rnn/strided_slice_3/stack_1%encoder_1/rnn/strided_slice_3/stack_2*
T0*
Index0*
new_axis_mask *
_output_shapes
: *
shrink_axis_mask*
ellipsis_mask *

begin_mask *
end_mask 
Z
encoder_1/rnn/stack_1/1Const*
dtype0*
_output_shapes
: *
value
B :�
�
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
�
encoder_1/rnn/zerosFillencoder_1/rnn/stack_1encoder_1/rnn/zeros/Const*
T0*(
_output_shapes
:����������
_
encoder_1/rnn/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
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
�
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
�
encoder_1/rnn/TensorArrayTensorArrayV3encoder_1/rnn/strided_slice_2*
dynamic_size( *
clear_after_read(*
_output_shapes

::*
element_shape:*
dtype0*9
tensor_array_name$"encoder_1/rnn/dynamic_rnn/output_0
�
encoder_1/rnn/TensorArray_1TensorArrayV3encoder_1/rnn/strided_slice_2*8
tensor_array_name#!encoder_1/rnn/dynamic_rnn/input_0*
dtype0*
element_shape:*
_output_shapes

::*
dynamic_size( *
clear_after_read(
{
&encoder_1/rnn/TensorArrayUnstack/ShapeConst*
dtype0*
_output_shapes
:*!
valueB"�   �   �   
~
4encoder_1/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
.encoder_1/rnn/TensorArrayUnstack/strided_sliceStridedSlice&encoder_1/rnn/TensorArrayUnstack/Shape4encoder_1/rnn/TensorArrayUnstack/strided_slice/stack6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_16encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask*
T0*
Index0*
end_mask *
_output_shapes
: *
ellipsis_mask *

begin_mask 
n
,encoder_1/rnn/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
n
,encoder_1/rnn/TensorArrayUnstack/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
�
&encoder_1/rnn/TensorArrayUnstack/rangeRange,encoder_1/rnn/TensorArrayUnstack/range/start.encoder_1/rnn/TensorArrayUnstack/strided_slice,encoder_1/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������*

Tidx0
�
Hencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3encoder_1/rnn/TensorArray_1&encoder_1/rnn/TensorArrayUnstack/rangeencoder_1/transposeencoder_1/rnn/TensorArray_1:1*
_output_shapes
: *.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
T0
�
encoder_1/rnn/while/EnterEnterencoder_1/rnn/time*
is_constant( *
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
�
encoder_1/rnn/while/Enter_1Enterencoder_1/rnn/TensorArray:1*
parallel_iterations *
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
�
encoder_1/rnn/while/Enter_2Enterencoder_1/initial_state_0_tiled*
is_constant( *(
_output_shapes
:����������*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
�
encoder_1/rnn/while/Enter_3Enterencoder_1/initial_state_1_tiled*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:����������*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
encoder_1/rnn/while/Enter_4Enterencoder_1/initial_state_2_tiled*
parallel_iterations *
T0*(
_output_shapes
:����������*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
�
encoder_1/rnn/while/Enter_5Enterencoder_1/initial_state_3_tiled*(
_output_shapes
:����������*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
�
encoder_1/rnn/while/MergeMergeencoder_1/rnn/while/Enter!encoder_1/rnn/while/NextIteration*
N*
T0*
_output_shapes
: : 
�
encoder_1/rnn/while/Merge_1Mergeencoder_1/rnn/while/Enter_1#encoder_1/rnn/while/NextIteration_1*
N*
T0*
_output_shapes
:: 
�
encoder_1/rnn/while/Merge_2Mergeencoder_1/rnn/while/Enter_2#encoder_1/rnn/while/NextIteration_2**
_output_shapes
:����������: *
N*
T0
�
encoder_1/rnn/while/Merge_3Mergeencoder_1/rnn/while/Enter_3#encoder_1/rnn/while/NextIteration_3**
_output_shapes
:����������: *
N*
T0
�
encoder_1/rnn/while/Merge_4Mergeencoder_1/rnn/while/Enter_4#encoder_1/rnn/while/NextIteration_4**
_output_shapes
:����������: *
N*
T0
�
encoder_1/rnn/while/Merge_5Mergeencoder_1/rnn/while/Enter_5#encoder_1/rnn/while/NextIteration_5**
_output_shapes
:����������: *
T0*
N
�
encoder_1/rnn/while/Less/EnterEnterencoder_1/rnn/strided_slice_2*
parallel_iterations *
T0*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(
|
encoder_1/rnn/while/LessLessencoder_1/rnn/while/Mergeencoder_1/rnn/while/Less/Enter*
T0*
_output_shapes
: 
Z
encoder_1/rnn/while/LoopCondLoopCondencoder_1/rnn/while/Less*
_output_shapes
: 
�
encoder_1/rnn/while/SwitchSwitchencoder_1/rnn/while/Mergeencoder_1/rnn/while/LoopCond*,
_class"
 loc:@encoder_1/rnn/while/Merge*
_output_shapes
: : *
T0
�
encoder_1/rnn/while/Switch_1Switchencoder_1/rnn/while/Merge_1encoder_1/rnn/while/LoopCond*
_output_shapes

::*.
_class$
" loc:@encoder_1/rnn/while/Merge_1*
T0
�
encoder_1/rnn/while/Switch_2Switchencoder_1/rnn/while/Merge_2encoder_1/rnn/while/LoopCond*
T0*<
_output_shapes*
(:����������:����������*.
_class$
" loc:@encoder_1/rnn/while/Merge_2
�
encoder_1/rnn/while/Switch_3Switchencoder_1/rnn/while/Merge_3encoder_1/rnn/while/LoopCond*
T0*.
_class$
" loc:@encoder_1/rnn/while/Merge_3*<
_output_shapes*
(:����������:����������
�
encoder_1/rnn/while/Switch_4Switchencoder_1/rnn/while/Merge_4encoder_1/rnn/while/LoopCond*
T0*<
_output_shapes*
(:����������:����������*.
_class$
" loc:@encoder_1/rnn/while/Merge_4
�
encoder_1/rnn/while/Switch_5Switchencoder_1/rnn/while/Merge_5encoder_1/rnn/while/LoopCond*.
_class$
" loc:@encoder_1/rnn/while/Merge_5*<
_output_shapes*
(:����������:����������*
T0
g
encoder_1/rnn/while/IdentityIdentityencoder_1/rnn/while/Switch:1*
_output_shapes
: *
T0
m
encoder_1/rnn/while/Identity_1Identityencoder_1/rnn/while/Switch_1:1*
_output_shapes
:*
T0
}
encoder_1/rnn/while/Identity_2Identityencoder_1/rnn/while/Switch_2:1*(
_output_shapes
:����������*
T0
}
encoder_1/rnn/while/Identity_3Identityencoder_1/rnn/while/Switch_3:1*(
_output_shapes
:����������*
T0
}
encoder_1/rnn/while/Identity_4Identityencoder_1/rnn/while/Switch_4:1*(
_output_shapes
:����������*
T0
}
encoder_1/rnn/while/Identity_5Identityencoder_1/rnn/while/Switch_5:1*(
_output_shapes
:����������*
T0
�
+encoder_1/rnn/while/TensorArrayReadV3/EnterEnterencoder_1/rnn/TensorArray_1*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
-encoder_1/rnn/while/TensorArrayReadV3/Enter_1EnterHencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
T0*
is_constant(
�
%encoder_1/rnn/while/TensorArrayReadV3TensorArrayReadV3+encoder_1/rnn/while/TensorArrayReadV3/Enterencoder_1/rnn/while/Identity-encoder_1/rnn/while/TensorArrayReadV3/Enter_1*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
dtype0* 
_output_shapes
:
��
�
Tencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB"      *
_output_shapes
:*
dtype0
�
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/minConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB
 *
ף�*
_output_shapes
: *
dtype0
�
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/maxConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB
 *
ף=*
dtype0*
_output_shapes
: 
�
\encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/RandomUniformRandomUniformTencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/shape*

seed{*
seed2�*
dtype0*
T0* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
�
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/subSubRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/maxRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/min*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
_output_shapes
: *
T0
�
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/mulMul\encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/RandomUniformRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
�
Nencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniformAddRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/mulRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/min* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0
�
3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
VariableV2*
	container *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
shared_name * 
_output_shapes
:
��*
shape:
��
�
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AssignAssign3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsNencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform* 
_output_shapes
:
��*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0*
use_locking(
�
8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/readIdentity3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
��*
T0
�
Iencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axisConst^encoder_1/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
�
Dencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concatConcatV2%encoder_1/rnn/while/TensorArrayReadV3encoder_1/rnn/while/Identity_3Iencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*

Tidx0*
T0*
N* 
_output_shapes
:
��
�
Jencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
��*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
Dencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMulMatMulDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concatJencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a( 
�
Dencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Initializer/ConstConst*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
valueB�*    *
dtype0*
_output_shapes	
:�
�
2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
VariableV2*
	container *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:�*
shape:�*
shared_name 
�
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AssignAssign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesDencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
�
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/readIdentity2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:�*
T0
�
Aencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/EnterEnter7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read*
is_constant(*
_output_shapes	
:�*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
�
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAddBiasAddDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMulAencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter* 
_output_shapes
:
��*
data_formatNHWC*
T0
�
Cencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dimConst^encoder_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/splitSplitCencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd*
T0*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split
�
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add/yConst^encoder_1/rnn/while/Identity*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/addAdd;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add/y*
T0* 
_output_shapes
:
��
�
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/SigmoidSigmoid7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add*
T0* 
_output_shapes
:
��
�
7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mulMul;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoidencoder_1/rnn/while/Identity_2* 
_output_shapes
:
��*
T0
�
=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1Sigmoid9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split* 
_output_shapes
:
��*
T0
�
8encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/TanhTanh;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:1* 
_output_shapes
:
��*
T0
�
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1Mul=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_18encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh* 
_output_shapes
:
��*
T0
�
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1Add7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1*
T0* 
_output_shapes
:
��
�
=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2Sigmoid;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:3*
T0* 
_output_shapes
:
��
�
:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1Tanh9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1*
T0* 
_output_shapes
:
��
�
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2Mul=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0* 
_output_shapes
:
��
�
Tencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB"      *
_output_shapes
:*
dtype0
�
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB
 *
ף�
�
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB
 *
ף=
�
\encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/RandomUniformRandomUniformTencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/shape*
T0* 
_output_shapes
:
��*

seed{*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
seed2�
�
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/subSubRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/maxRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
_output_shapes
: 
�
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/mulMul\encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/RandomUniformRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/sub* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0
�
Nencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniformAddRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/mulRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/min*
T0* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
�
3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
VariableV2*
shape:
��* 
_output_shapes
:
��*
shared_name *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
	container 
�
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AssignAssign3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsNencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
�
8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/readIdentity3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
��*
T0
�
Iencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axisConst^encoder_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Dencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concatConcatV29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2encoder_1/rnn/while/Identity_5Iencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis* 
_output_shapes
:
��*
T0*

Tidx0*
N
�
Jencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read*
parallel_iterations *
T0* 
_output_shapes
:
��*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(
�
Dencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMulMatMulDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concatJencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter*
transpose_b( * 
_output_shapes
:
��*
transpose_a( *
T0
�
Dencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Initializer/ConstConst*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
valueB�*    *
dtype0*
_output_shapes	
:�
�
2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
VariableV2*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AssignAssign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesDencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Initializer/Const*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:�*
T0*
validate_shape(*
use_locking(
�
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/readIdentity2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
_output_shapes	
:�
�
Aencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/EnterEnter7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:�*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAddBiasAddDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMulAencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter*
data_formatNHWC*
T0* 
_output_shapes
:
��
�
Cencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dimConst^encoder_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/splitSplitCencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split*
T0
�
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add/yConst^encoder_1/rnn/while/Identity*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/addAdd;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:29encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add/y* 
_output_shapes
:
��*
T0
�
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/SigmoidSigmoid7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add* 
_output_shapes
:
��*
T0
�
7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mulMul;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoidencoder_1/rnn/while/Identity_4* 
_output_shapes
:
��*
T0
�
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1Sigmoid9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split* 
_output_shapes
:
��*
T0
�
8encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/TanhTanh;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:1* 
_output_shapes
:
��*
T0
�
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1Mul=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_18encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh* 
_output_shapes
:
��*
T0
�
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1Add7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1* 
_output_shapes
:
��*
T0
�
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2Sigmoid;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:3*
T0* 
_output_shapes
:
��
�
:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1Tanh9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1* 
_output_shapes
:
��*
T0
�
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2Mul=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1* 
_output_shapes
:
��*
T0
�
&encoder_1/rnn/while/GreaterEqual/EnterEnterencoder_1/rnn/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:�*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
 encoder_1/rnn/while/GreaterEqualGreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:�
�
 encoder_1/rnn/while/Select/EnterEnterencoder_1/rnn/zeros*
T0*
is_constant(*
parallel_iterations *(
_output_shapes
:����������*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
encoder_1/rnn/while/SelectSelect encoder_1/rnn/while/GreaterEqual encoder_1/rnn/while/Select/Enter9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0* 
_output_shapes
:
��
�
"encoder_1/rnn/while/GreaterEqual_1GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:�
�
encoder_1/rnn/while/Select_1Select"encoder_1/rnn/while/GreaterEqual_1encoder_1/rnn/while/Identity_29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1*
T0* 
_output_shapes
:
��
�
"encoder_1/rnn/while/GreaterEqual_2GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
_output_shapes	
:�*
T0
�
encoder_1/rnn/while/Select_2Select"encoder_1/rnn/while/GreaterEqual_2encoder_1/rnn/while/Identity_39encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2* 
_output_shapes
:
��*
T0
�
"encoder_1/rnn/while/GreaterEqual_3GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
_output_shapes	
:�*
T0
�
encoder_1/rnn/while/Select_3Select"encoder_1/rnn/while/GreaterEqual_3encoder_1/rnn/while/Identity_49encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1*
T0* 
_output_shapes
:
��
�
"encoder_1/rnn/while/GreaterEqual_4GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:�
�
encoder_1/rnn/while/Select_4Select"encoder_1/rnn/while/GreaterEqual_4encoder_1/rnn/while/Identity_59encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2* 
_output_shapes
:
��*
T0
�
=encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterencoder_1/rnn/TensorArray*
T0*,
_class"
 loc:@encoder_1/rnn/TensorArray*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
7encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3=encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterencoder_1/rnn/while/Identityencoder_1/rnn/while/Selectencoder_1/rnn/while/Identity_1*
T0*
_output_shapes
: *,
_class"
 loc:@encoder_1/rnn/TensorArray
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
!encoder_1/rnn/while/NextIterationNextIterationencoder_1/rnn/while/add*
T0*
_output_shapes
: 
�
#encoder_1/rnn/while/NextIteration_1NextIteration7encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
}
#encoder_1/rnn/while/NextIteration_2NextIterationencoder_1/rnn/while/Select_1* 
_output_shapes
:
��*
T0
}
#encoder_1/rnn/while/NextIteration_3NextIterationencoder_1/rnn/while/Select_2* 
_output_shapes
:
��*
T0
}
#encoder_1/rnn/while/NextIteration_4NextIterationencoder_1/rnn/while/Select_3*
T0* 
_output_shapes
:
��
}
#encoder_1/rnn/while/NextIteration_5NextIterationencoder_1/rnn/while/Select_4*
T0* 
_output_shapes
:
��
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
encoder_1/rnn/while/Exit_2Exitencoder_1/rnn/while/Switch_2*
T0*(
_output_shapes
:����������
s
encoder_1/rnn/while/Exit_3Exitencoder_1/rnn/while/Switch_3*(
_output_shapes
:����������*
T0
s
encoder_1/rnn/while/Exit_4Exitencoder_1/rnn/while/Switch_4*
T0*(
_output_shapes
:����������
s
encoder_1/rnn/while/Exit_5Exitencoder_1/rnn/while/Switch_5*(
_output_shapes
:����������*
T0
�
0encoder_1/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3encoder_1/rnn/TensorArrayencoder_1/rnn/while/Exit_1*,
_class"
 loc:@encoder_1/rnn/TensorArray*
_output_shapes
: 
�
*encoder_1/rnn/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *
value	B : *,
_class"
 loc:@encoder_1/rnn/TensorArray
�
*encoder_1/rnn/TensorArrayStack/range/deltaConst*
value	B :*,
_class"
 loc:@encoder_1/rnn/TensorArray*
dtype0*
_output_shapes
: 
�
$encoder_1/rnn/TensorArrayStack/rangeRange*encoder_1/rnn/TensorArrayStack/range/start0encoder_1/rnn/TensorArrayStack/TensorArraySizeV3*encoder_1/rnn/TensorArrayStack/range/delta*

Tidx0*,
_class"
 loc:@encoder_1/rnn/TensorArray*#
_output_shapes
:���������
�
2encoder_1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3encoder_1/rnn/TensorArray$encoder_1/rnn/TensorArrayStack/rangeencoder_1/rnn/while/Exit_1*%
_output_shapes
:���*
dtype0*,
_class"
 loc:@encoder_1/rnn/TensorArray*
element_shape:
��
q
encoder_1/rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
�
encoder_1/rnn/transpose	Transpose2encoder_1/rnn/TensorArrayStack/TensorArrayGatherV3encoder_1/rnn/transpose/perm*
Tperm0*%
_output_shapes
:���*
T0
�
6output_projection/W/Initializer/truncated_normal/shapeConst*&
_class
loc:@output_projection/W*
valueB"�      *
dtype0*
_output_shapes
:
�
5output_projection/W/Initializer/truncated_normal/meanConst*&
_class
loc:@output_projection/W*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7output_projection/W/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *&
_class
loc:@output_projection/W*
valueB
 *���=
�
@output_projection/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6output_projection/W/Initializer/truncated_normal/shape*
_output_shapes
:	�*
dtype0*
seed2�*&
_class
loc:@output_projection/W*
T0*

seed{
�
4output_projection/W/Initializer/truncated_normal/mulMul@output_projection/W/Initializer/truncated_normal/TruncatedNormal7output_projection/W/Initializer/truncated_normal/stddev*
T0*&
_class
loc:@output_projection/W*
_output_shapes
:	�
�
0output_projection/W/Initializer/truncated_normalAdd4output_projection/W/Initializer/truncated_normal/mul5output_projection/W/Initializer/truncated_normal/mean*
T0*&
_class
loc:@output_projection/W*
_output_shapes
:	�
�
output_projection/W
VariableV2*
shared_name *&
_class
loc:@output_projection/W*
	container *
shape:	�*
dtype0*
_output_shapes
:	�
�
output_projection/W/AssignAssignoutput_projection/W0output_projection/W/Initializer/truncated_normal*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�*&
_class
loc:@output_projection/W
�
output_projection/W/readIdentityoutput_projection/W*
T0*&
_class
loc:@output_projection/W*
_output_shapes
:	�
�
%output_projection/b/Initializer/ConstConst*&
_class
loc:@output_projection/b*
valueB*���=*
dtype0*
_output_shapes
:
�
output_projection/b
VariableV2*
shape:*
_output_shapes
:*
shared_name *&
_class
loc:@output_projection/b*
dtype0*
	container 
�
output_projection/b/AssignAssignoutput_projection/b%output_projection/b/Initializer/Const*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
�
output_projection/b/readIdentityoutput_projection/b*
T0*&
_class
loc:@output_projection/b*
_output_shapes
:
�
"output_projection/xw_plus_b/MatMulMatMulencoder_1/rnn/while/Exit_4output_projection/W/read*
transpose_b( *'
_output_shapes
:���������*
transpose_a( *
T0
�
output_projection/xw_plus_bBiasAdd"output_projection/xw_plus_b/MatMuloutput_projection/b/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
s
output_projection/SoftmaxSoftmaxoutput_projection/xw_plus_b*'
_output_shapes
:���������*
T0
d
"output_projection/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
output_projection/ArgMaxArgMaxoutput_projection/xw_plus_b"output_projection/ArgMax/dimension*#
_output_shapes
:���������*
T0*

Tidx0
o

loss/ConstConst*
dtype0*
_output_shapes
:*1
value(B&"  �?��z?`�p?bX?�p}?  �?m�{>
j
loss/MulMuloutput_projection/xw_plus_b
loss/Const*'
_output_shapes
:���������*
T0
X
	loss/CastCastcond/Merge_1*

SrcT0*
_output_shapes
:	�*

DstT0
]
loss/logistic_loss/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
h
loss/logistic_loss/subSub
loss/Constloss/logistic_loss/sub/y*
_output_shapes
:*
T0
j
loss/logistic_loss/mulMulloss/logistic_loss/sub	loss/Cast*
_output_shapes
:	�*
T0
]
loss/logistic_loss/add/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
y
loss/logistic_loss/addAddloss/logistic_loss/add/xloss/logistic_loss/mul*
T0*
_output_shapes
:	�
_
loss/logistic_loss/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
p
loss/logistic_loss/sub_1Subloss/logistic_loss/sub_1/x	loss/Cast*
T0*
_output_shapes
:	�
m
loss/logistic_loss/mul_1Mulloss/logistic_loss/sub_1loss/Mul*
_output_shapes
:	�*
T0
Y
loss/logistic_loss/AbsAbsloss/Mul*'
_output_shapes
:���������*
T0
g
loss/logistic_loss/NegNegloss/logistic_loss/Abs*
T0*'
_output_shapes
:���������
g
loss/logistic_loss/ExpExploss/logistic_loss/Neg*
T0*'
_output_shapes
:���������
k
loss/logistic_loss/Log1pLog1ploss/logistic_loss/Exp*'
_output_shapes
:���������*
T0
[
loss/logistic_loss/Neg_1Negloss/Mul*
T0*'
_output_shapes
:���������
k
loss/logistic_loss/ReluReluloss/logistic_loss/Neg_1*
T0*'
_output_shapes
:���������
�
loss/logistic_loss/add_1Addloss/logistic_loss/Log1ploss/logistic_loss/Relu*'
_output_shapes
:���������*
T0
{
loss/logistic_loss/mul_2Mulloss/logistic_loss/addloss/logistic_loss/add_1*
_output_shapes
:	�*
T0
w
loss/logistic_lossAddloss/logistic_loss/mul_1loss/logistic_loss/mul_2*
T0*
_output_shapes
:	�
]
loss/Const_1Const*
dtype0*
_output_shapes
:*
valueB"       
o
loss/SumSumloss/logistic_lossloss/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
]
loss/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       
q
	loss/MeanMeanloss/logistic_lossloss/Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
[
accuracy/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
t
accuracy/ArgMaxArgMaxcond/Merge_1accuracy/ArgMax/dimension*
_output_shapes	
:�*
T0*

Tidx0
h
accuracy/EqualEqualoutput_projection/ArgMaxaccuracy/ArgMax*
_output_shapes	
:�*
T0	
Z
accuracy/CastCastaccuracy/Equal*

SrcT0
*
_output_shapes	
:�*

DstT0
X
accuracy/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
v
accuracy/accuracyMeanaccuracy/Castaccuracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
`
learning_rate/learning_rateConst*
valueB
 *�Q8*
dtype0*
_output_shapes
: 
Y
learning_rate/CastCastVariable/read*
_output_shapes
: *

DstT0*

SrcT0
Y
learning_rate/Cast_1/xConst*
dtype0*
_output_shapes
: *
value
B :�'
d
learning_rate/Cast_1Castlearning_rate/Cast_1/x*
_output_shapes
: *

DstT0*

SrcT0
[
learning_rate/Cast_2/xConst*
valueB
 *��u?*
_output_shapes
: *
dtype0
k
learning_rate/truedivRealDivlearning_rate/Castlearning_rate/Cast_1*
T0*
_output_shapes
: 
T
learning_rate/FloorFloorlearning_rate/truediv*
T0*
_output_shapes
: 
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
valueB"�      
T
gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
b
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
:	�
S
gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
�
gradients/f_count_1Entergradients/f_count*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
_output_shapes
: : *
T0*
N
l
gradients/SwitchSwitchgradients/Mergeencoder_1/rnn/while/LoopCond*
_output_shapes
: : *
T0
p
gradients/Add/yConst^encoder_1/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
_output_shapes
: *
T0
�
gradients/NextIterationNextIterationgradients/AddA^gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPushA^gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPushA^gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPushA^gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPush=^gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPushZ^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPushb^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPushe^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPushW^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPushY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPushZ^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPushg^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPushb^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPushe^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPushc^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPush*
_output_shapes
: *
T0
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
�
gradients/b_count_1Entergradients/f_count_2*
parallel_iterations *
T0*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
gradients/GreaterEqual/EnterEntergradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
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
�
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
dtype0*
_output_shapes
:*
valueB"�      
z
)gradients/loss/logistic_loss_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"�      
�
7gradients/loss/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/loss/logistic_loss_grad/Shape)gradients/loss/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
%gradients/loss/logistic_loss_grad/SumSumgradients/Fill7gradients/loss/logistic_loss_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
)gradients/loss/logistic_loss_grad/ReshapeReshape%gradients/loss/logistic_loss_grad/Sum'gradients/loss/logistic_loss_grad/Shape*
Tshape0*
_output_shapes
:	�*
T0
�
'gradients/loss/logistic_loss_grad/Sum_1Sumgradients/Fill9gradients/loss/logistic_loss_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
+gradients/loss/logistic_loss_grad/Reshape_1Reshape'gradients/loss/logistic_loss_grad/Sum_1)gradients/loss/logistic_loss_grad/Shape_1*
Tshape0*
_output_shapes
:	�*
T0
~
-gradients/loss/logistic_loss/mul_1_grad/ShapeConst*
valueB"�      *
_output_shapes
:*
dtype0
w
/gradients/loss/logistic_loss/mul_1_grad/Shape_1Shapeloss/Mul*
T0*
_output_shapes
:*
out_type0
�
=gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/mul_1_grad/Shape/gradients/loss/logistic_loss/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
+gradients/loss/logistic_loss/mul_1_grad/mulMul)gradients/loss/logistic_loss_grad/Reshapeloss/Mul*
T0*
_output_shapes
:	�
�
+gradients/loss/logistic_loss/mul_1_grad/SumSum+gradients/loss/logistic_loss/mul_1_grad/mul=gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
/gradients/loss/logistic_loss/mul_1_grad/ReshapeReshape+gradients/loss/logistic_loss/mul_1_grad/Sum-gradients/loss/logistic_loss/mul_1_grad/Shape*
T0*
_output_shapes
:	�*
Tshape0
�
-gradients/loss/logistic_loss/mul_1_grad/mul_1Mulloss/logistic_loss/sub_1)gradients/loss/logistic_loss_grad/Reshape*
_output_shapes
:	�*
T0
�
-gradients/loss/logistic_loss/mul_1_grad/Sum_1Sum-gradients/loss/logistic_loss/mul_1_grad/mul_1?gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
1gradients/loss/logistic_loss/mul_1_grad/Reshape_1Reshape-gradients/loss/logistic_loss/mul_1_grad/Sum_1/gradients/loss/logistic_loss/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
~
-gradients/loss/logistic_loss/mul_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"�      
�
/gradients/loss/logistic_loss/mul_2_grad/Shape_1Shapeloss/logistic_loss/add_1*
T0*
_output_shapes
:*
out_type0
�
=gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/mul_2_grad/Shape/gradients/loss/logistic_loss/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
+gradients/loss/logistic_loss/mul_2_grad/mulMul+gradients/loss/logistic_loss_grad/Reshape_1loss/logistic_loss/add_1*
T0*
_output_shapes
:	�
�
+gradients/loss/logistic_loss/mul_2_grad/SumSum+gradients/loss/logistic_loss/mul_2_grad/mul=gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
/gradients/loss/logistic_loss/mul_2_grad/ReshapeReshape+gradients/loss/logistic_loss/mul_2_grad/Sum-gradients/loss/logistic_loss/mul_2_grad/Shape*
_output_shapes
:	�*
Tshape0*
T0
�
-gradients/loss/logistic_loss/mul_2_grad/mul_1Mulloss/logistic_loss/add+gradients/loss/logistic_loss_grad/Reshape_1*
_output_shapes
:	�*
T0
�
-gradients/loss/logistic_loss/mul_2_grad/Sum_1Sum-gradients/loss/logistic_loss/mul_2_grad/mul_1?gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
1gradients/loss/logistic_loss/mul_2_grad/Reshape_1Reshape-gradients/loss/logistic_loss/mul_2_grad/Sum_1/gradients/loss/logistic_loss/mul_2_grad/Shape_1*'
_output_shapes
:���������*
Tshape0*
T0
�
-gradients/loss/logistic_loss/add_1_grad/ShapeShapeloss/logistic_loss/Log1p*
T0*
_output_shapes
:*
out_type0
�
/gradients/loss/logistic_loss/add_1_grad/Shape_1Shapeloss/logistic_loss/Relu*
out_type0*
_output_shapes
:*
T0
�
=gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/add_1_grad/Shape/gradients/loss/logistic_loss/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
+gradients/loss/logistic_loss/add_1_grad/SumSum1gradients/loss/logistic_loss/mul_2_grad/Reshape_1=gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
/gradients/loss/logistic_loss/add_1_grad/ReshapeReshape+gradients/loss/logistic_loss/add_1_grad/Sum-gradients/loss/logistic_loss/add_1_grad/Shape*
T0*'
_output_shapes
:���������*
Tshape0
�
-gradients/loss/logistic_loss/add_1_grad/Sum_1Sum1gradients/loss/logistic_loss/mul_2_grad/Reshape_1?gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
1gradients/loss/logistic_loss/add_1_grad/Reshape_1Reshape-gradients/loss/logistic_loss/add_1_grad/Sum_1/gradients/loss/logistic_loss/add_1_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
�
-gradients/loss/logistic_loss/Log1p_grad/add/xConst0^gradients/loss/logistic_loss/add_1_grad/Reshape*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
+gradients/loss/logistic_loss/Log1p_grad/addAdd-gradients/loss/logistic_loss/Log1p_grad/add/xloss/logistic_loss/Exp*'
_output_shapes
:���������*
T0
�
2gradients/loss/logistic_loss/Log1p_grad/Reciprocal
Reciprocal+gradients/loss/logistic_loss/Log1p_grad/add*
T0*'
_output_shapes
:���������
�
+gradients/loss/logistic_loss/Log1p_grad/mulMul/gradients/loss/logistic_loss/add_1_grad/Reshape2gradients/loss/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:���������*
T0
�
/gradients/loss/logistic_loss/Relu_grad/ReluGradReluGrad1gradients/loss/logistic_loss/add_1_grad/Reshape_1loss/logistic_loss/Relu*'
_output_shapes
:���������*
T0
�
)gradients/loss/logistic_loss/Exp_grad/mulMul+gradients/loss/logistic_loss/Log1p_grad/mulloss/logistic_loss/Exp*
T0*'
_output_shapes
:���������
�
+gradients/loss/logistic_loss/Neg_1_grad/NegNeg/gradients/loss/logistic_loss/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������
�
)gradients/loss/logistic_loss/Neg_grad/NegNeg)gradients/loss/logistic_loss/Exp_grad/mul*'
_output_shapes
:���������*
T0
n
*gradients/loss/logistic_loss/Abs_grad/SignSignloss/Mul*'
_output_shapes
:���������*
T0
�
)gradients/loss/logistic_loss/Abs_grad/mulMul)gradients/loss/logistic_loss/Neg_grad/Neg*gradients/loss/logistic_loss/Abs_grad/Sign*
T0*'
_output_shapes
:���������
�
gradients/AddNAddN1gradients/loss/logistic_loss/mul_1_grad/Reshape_1+gradients/loss/logistic_loss/Neg_1_grad/Neg)gradients/loss/logistic_loss/Abs_grad/mul*
T0*D
_class:
86loc:@gradients/loss/logistic_loss/mul_1_grad/Reshape_1*
N*'
_output_shapes
:���������
x
gradients/loss/Mul_grad/ShapeShapeoutput_projection/xw_plus_b*
out_type0*
_output_shapes
:*
T0
i
gradients/loss/Mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
-gradients/loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/Mul_grad/Shapegradients/loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
p
gradients/loss/Mul_grad/mulMulgradients/AddN
loss/Const*
T0*'
_output_shapes
:���������
�
gradients/loss/Mul_grad/SumSumgradients/loss/Mul_grad/mul-gradients/loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/loss/Mul_grad/ReshapeReshapegradients/loss/Mul_grad/Sumgradients/loss/Mul_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/loss/Mul_grad/mul_1Muloutput_projection/xw_plus_bgradients/AddN*'
_output_shapes
:���������*
T0
�
gradients/loss/Mul_grad/Sum_1Sumgradients/loss/Mul_grad/mul_1/gradients/loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
!gradients/loss/Mul_grad/Reshape_1Reshapegradients/loss/Mul_grad/Sum_1gradients/loss/Mul_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
�
6gradients/output_projection/xw_plus_b_grad/BiasAddGradBiasAddGradgradients/loss/Mul_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
�
8gradients/output_projection/xw_plus_b/MatMul_grad/MatMulMatMulgradients/loss/Mul_grad/Reshapeoutput_projection/W/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1MatMulencoder_1/rnn/while/Exit_4gradients/loss/Mul_grad/Reshape*
transpose_b( *
T0*
_output_shapes
:	�*
transpose_a(
`
gradients/zeros_like	ZerosLikeencoder_1/rnn/while/Exit_1*
T0*
_output_shapes
:
r
gradients/zeros_like_1	ZerosLikeencoder_1/rnn/while/Exit_2*
T0*(
_output_shapes
:����������
r
gradients/zeros_like_2	ZerosLikeencoder_1/rnn/while/Exit_3*
T0*(
_output_shapes
:����������
r
gradients/zeros_like_3	ZerosLikeencoder_1/rnn/while/Exit_5*(
_output_shapes
:����������*
T0
�
0gradients/encoder_1/rnn/while/Exit_4_grad/b_exitEnter8gradients/output_projection/xw_plus_b/MatMul_grad/MatMul*
is_constant( *(
_output_shapes
:����������*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
�
0gradients/encoder_1/rnn/while/Exit_1_grad/b_exitEntergradients/zeros_like*
is_constant( *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
�
0gradients/encoder_1/rnn/while/Exit_2_grad/b_exitEntergradients/zeros_like_1*(
_output_shapes
:����������*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
�
0gradients/encoder_1/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_like_2*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:����������*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
0gradients/encoder_1/rnn/while/Exit_5_grad/b_exitEntergradients/zeros_like_3*
is_constant( *(
_output_shapes
:����������*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
�
4gradients/encoder_1/rnn/while/Switch_4_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_4_grad/b_exit;gradients/encoder_1/rnn/while/Switch_4_grad_1/NextIteration**
_output_shapes
:����������: *
T0*
N
�
4gradients/encoder_1/rnn/while/Switch_2_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_2_grad/b_exit;gradients/encoder_1/rnn/while/Switch_2_grad_1/NextIteration*
N*
T0**
_output_shapes
:����������: 
�
4gradients/encoder_1/rnn/while/Switch_3_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_3_grad/b_exit;gradients/encoder_1/rnn/while/Switch_3_grad_1/NextIteration**
_output_shapes
:����������: *
N*
T0
�
4gradients/encoder_1/rnn/while/Switch_5_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_5_grad/b_exit;gradients/encoder_1/rnn/while/Switch_5_grad_1/NextIteration*
N*
T0**
_output_shapes
:����������: 
�
1gradients/encoder_1/rnn/while/Merge_4_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*
T0*4
_output_shapes"
 :����������:
��*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_4_grad/b_switch
�
1gradients/encoder_1/rnn/while/Merge_2_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_2_grad/b_switch*4
_output_shapes"
 :����������:
��*
T0
�
1gradients/encoder_1/rnn/while/Merge_3_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*
T0*4
_output_shapes"
 :����������:
��*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_3_grad/b_switch
�
1gradients/encoder_1/rnn/while/Merge_5_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_5_grad/b_switchgradients/b_count_2*4
_output_shapes"
 :����������:
��*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_5_grad/b_switch*
T0
�
/gradients/encoder_1/rnn/while/Enter_4_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_4_grad/Switch*(
_output_shapes
:����������*
T0
�
/gradients/encoder_1/rnn/while/Enter_2_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_2_grad/Switch*
T0*(
_output_shapes
:����������
�
/gradients/encoder_1/rnn/while/Enter_3_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_3_grad/Switch*(
_output_shapes
:����������*
T0
�
/gradients/encoder_1/rnn/while/Enter_5_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_5_grad/Switch*
T0*(
_output_shapes
:����������
�
4gradients/encoder_1/initial_state_2_tiled_grad/ShapeConst*
valueB"   �   *
dtype0*
_output_shapes
:
�
4gradients/encoder_1/initial_state_2_tiled_grad/stackPack)encoder_1/initial_state_2_tiled/multiples4gradients/encoder_1/initial_state_2_tiled_grad/Shape*
_output_shapes

:*
N*

axis *
T0
�
=gradients/encoder_1/initial_state_2_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_2_tiled_grad/stack*
T0*
_output_shapes
: 
�
>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub/yConst*
_output_shapes
: *
dtype0*
value	B :
�
<gradients/encoder_1/initial_state_2_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_2_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub/y*
_output_shapes
: *
T0
�
Dgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
Dgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
>gradients/encoder_1/initial_state_2_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_2_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/delta*
_output_shapes
:*

Tidx0
�
>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_2_tiled_grad/transpose/Range*
_output_shapes
:*
T0
�
8gradients/encoder_1/initial_state_2_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_2_tiled_grad/stack>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub_1*
Tperm0*
_output_shapes

:*
T0
�
<gradients/encoder_1/initial_state_2_tiled_grad/Reshape/shapeConst*
valueB:
���������*
_output_shapes
:*
dtype0
�
6gradients/encoder_1/initial_state_2_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_2_tiled_grad/transpose<gradients/encoder_1/initial_state_2_tiled_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
u
3gradients/encoder_1/initial_state_2_tiled_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
|
:gradients/encoder_1/initial_state_2_tiled_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
|
:gradients/encoder_1/initial_state_2_tiled_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
4gradients/encoder_1/initial_state_2_tiled_grad/rangeRange:gradients/encoder_1/initial_state_2_tiled_grad/range/start3gradients/encoder_1/initial_state_2_tiled_grad/Size:gradients/encoder_1/initial_state_2_tiled_grad/range/delta*
_output_shapes
:*

Tidx0
�
8gradients/encoder_1/initial_state_2_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_4_grad/Exit6gradients/encoder_1/initial_state_2_tiled_grad/Reshape*
T0*J
_output_shapes8
6:4������������������������������������*
Tshape0
�
2gradients/encoder_1/initial_state_2_tiled_grad/SumSum8gradients/encoder_1/initial_state_2_tiled_grad/Reshape_14gradients/encoder_1/initial_state_2_tiled_grad/range*
_output_shapes
:	�*
T0*
	keep_dims( *

Tidx0
�
<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *1
_class'
%#loc:@encoder_1/rnn/while/Identity_4
�
?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
@gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_4^gradients/Add*
_output_shapes
:*
swap_memory( *1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
T0
�
Hgradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*(
_output_shapes
:����������
�
=gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/b_syncControlTrigger@^gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopa^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopd^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopa^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopd^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopb^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop
�
6gradients/encoder_1/rnn/while/Select_3_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop*
T0*(
_output_shapes
:����������
�
8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*

stack_name *
_output_shapes
:
�
;gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
<gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_3^gradients/Add*
_output_shapes
:*
swap_memory( *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
T0

�
Dgradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
;gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
_output_shapes	
:�
�
2gradients/encoder_1/rnn/while/Select_3_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_4_grad/Switch:16gradients/encoder_1/rnn/while/Select_3_grad/zeros_like* 
_output_shapes
:
��*
T0
�
4gradients/encoder_1/rnn/while/Select_3_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_3_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_4_grad/Switch:1*
T0* 
_output_shapes
:
��
�
4gradients/encoder_1/initial_state_0_tiled_grad/ShapeConst*
valueB"   �   *
dtype0*
_output_shapes
:
�
4gradients/encoder_1/initial_state_0_tiled_grad/stackPack)encoder_1/initial_state_0_tiled/multiples4gradients/encoder_1/initial_state_0_tiled_grad/Shape*

axis *
_output_shapes

:*
T0*
N
�
=gradients/encoder_1/initial_state_0_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_0_tiled_grad/stack*
_output_shapes
: *
T0
�
>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
<gradients/encoder_1/initial_state_0_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_0_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub/y*
T0*
_output_shapes
: 
�
Dgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 
�
Dgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
�
>gradients/encoder_1/initial_state_0_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_0_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/delta*
_output_shapes
:*

Tidx0
�
>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_0_tiled_grad/transpose/Range*
_output_shapes
:*
T0
�
8gradients/encoder_1/initial_state_0_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_0_tiled_grad/stack>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub_1*
Tperm0*
_output_shapes

:*
T0
�
<gradients/encoder_1/initial_state_0_tiled_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������
�
6gradients/encoder_1/initial_state_0_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_0_tiled_grad/transpose<gradients/encoder_1/initial_state_0_tiled_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
u
3gradients/encoder_1/initial_state_0_tiled_grad/SizeConst*
value	B :*
_output_shapes
: *
dtype0
|
:gradients/encoder_1/initial_state_0_tiled_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
|
:gradients/encoder_1/initial_state_0_tiled_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
�
4gradients/encoder_1/initial_state_0_tiled_grad/rangeRange:gradients/encoder_1/initial_state_0_tiled_grad/range/start3gradients/encoder_1/initial_state_0_tiled_grad/Size:gradients/encoder_1/initial_state_0_tiled_grad/range/delta*

Tidx0*
_output_shapes
:
�
8gradients/encoder_1/initial_state_0_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_2_grad/Exit6gradients/encoder_1/initial_state_0_tiled_grad/Reshape*J
_output_shapes8
6:4������������������������������������*
Tshape0*
T0
�
2gradients/encoder_1/initial_state_0_tiled_grad/SumSum8gradients/encoder_1/initial_state_0_tiled_grad/Reshape_14gradients/encoder_1/initial_state_0_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	�
�
<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_accStack*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*

stack_name *
_output_shapes
:
�
?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
@gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_2^gradients/Add*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*
_output_shapes
:*
swap_memory( 
�
Hgradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*
T0
�
?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*(
_output_shapes
:����������
�
6gradients/encoder_1/rnn/while/Select_1_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop*
T0*(
_output_shapes
:����������
�
8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_accStack*
	elem_type0
*
_output_shapes
:*

stack_name *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1
�
;gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc*
is_constant(*
T0*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
�
<gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_1^gradients/Add*
_output_shapes
:*
swap_memory( *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
T0

�
Dgradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1
�
;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
_output_shapes	
:�
�
2gradients/encoder_1/rnn/while/Select_1_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_2_grad/Switch:16gradients/encoder_1/rnn/while/Select_1_grad/zeros_like*
T0* 
_output_shapes
:
��
�
4gradients/encoder_1/rnn/while/Select_1_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_1_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_2_grad/Switch:1*
T0* 
_output_shapes
:
��
�
4gradients/encoder_1/initial_state_1_tiled_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   �   
�
4gradients/encoder_1/initial_state_1_tiled_grad/stackPack)encoder_1/initial_state_1_tiled/multiples4gradients/encoder_1/initial_state_1_tiled_grad/Shape*
T0*

axis *
N*
_output_shapes

:
�
=gradients/encoder_1/initial_state_1_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_1_tiled_grad/stack*
_output_shapes
: *
T0
�
>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub/yConst*
value	B :*
_output_shapes
: *
dtype0
�
<gradients/encoder_1/initial_state_1_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_1_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub/y*
T0*
_output_shapes
: 
�
Dgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 
�
Dgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
>gradients/encoder_1/initial_state_1_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_1_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/delta*

Tidx0*
_output_shapes
:
�
>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_1_tiled_grad/transpose/Range*
T0*
_output_shapes
:
�
8gradients/encoder_1/initial_state_1_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_1_tiled_grad/stack>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub_1*
Tperm0*
_output_shapes

:*
T0
�
<gradients/encoder_1/initial_state_1_tiled_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������
�
6gradients/encoder_1/initial_state_1_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_1_tiled_grad/transpose<gradients/encoder_1/initial_state_1_tiled_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
u
3gradients/encoder_1/initial_state_1_tiled_grad/SizeConst*
dtype0*
_output_shapes
: *
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
�
4gradients/encoder_1/initial_state_1_tiled_grad/rangeRange:gradients/encoder_1/initial_state_1_tiled_grad/range/start3gradients/encoder_1/initial_state_1_tiled_grad/Size:gradients/encoder_1/initial_state_1_tiled_grad/range/delta*
_output_shapes
:*

Tidx0
�
8gradients/encoder_1/initial_state_1_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_3_grad/Exit6gradients/encoder_1/initial_state_1_tiled_grad/Reshape*
Tshape0*J
_output_shapes8
6:4������������������������������������*
T0
�
2gradients/encoder_1/initial_state_1_tiled_grad/SumSum8gradients/encoder_1/initial_state_1_tiled_grad/Reshape_14gradients/encoder_1/initial_state_1_tiled_grad/range*
_output_shapes
:	�*
T0*
	keep_dims( *

Tidx0
�
<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_accStack*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*
_output_shapes
:*

stack_name 
�
?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3
�
@gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_3^gradients/Add*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*
_output_shapes
:*
swap_memory( 
�
Hgradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*(
_output_shapes
:����������
�
6gradients/encoder_1/rnn/while/Select_2_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop*(
_output_shapes
:����������*
T0
�
8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
_output_shapes
:*

stack_name 
�
;gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
T0
�
<gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_2^gradients/Add*
T0
*
_output_shapes
:*
swap_memory( *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2
�
Dgradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc*
T0*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*
_output_shapes	
:�*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2
�
2gradients/encoder_1/rnn/while/Select_2_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_3_grad/Switch:16gradients/encoder_1/rnn/while/Select_2_grad/zeros_like* 
_output_shapes
:
��*
T0
�
4gradients/encoder_1/rnn/while/Select_2_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_2_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_3_grad/Switch:1* 
_output_shapes
:
��*
T0
�
4gradients/encoder_1/initial_state_3_tiled_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   �   
�
4gradients/encoder_1/initial_state_3_tiled_grad/stackPack)encoder_1/initial_state_3_tiled/multiples4gradients/encoder_1/initial_state_3_tiled_grad/Shape*
T0*

axis *
N*
_output_shapes

:
�
=gradients/encoder_1/initial_state_3_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_3_tiled_grad/stack*
T0*
_output_shapes
: 
�
>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub/yConst*
value	B :*
_output_shapes
: *
dtype0
�
<gradients/encoder_1/initial_state_3_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_3_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub/y*
T0*
_output_shapes
: 
�
Dgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
Dgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
>gradients/encoder_1/initial_state_3_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_3_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/delta*
_output_shapes
:*

Tidx0
�
>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub_1Sub<gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub>gradients/encoder_1/initial_state_3_tiled_grad/transpose/Range*
_output_shapes
:*
T0
�
8gradients/encoder_1/initial_state_3_tiled_grad/transpose	Transpose4gradients/encoder_1/initial_state_3_tiled_grad/stack>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub_1*
Tperm0*
_output_shapes

:*
T0
�
<gradients/encoder_1/initial_state_3_tiled_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������
�
6gradients/encoder_1/initial_state_3_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_3_tiled_grad/transpose<gradients/encoder_1/initial_state_3_tiled_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
u
3gradients/encoder_1/initial_state_3_tiled_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
|
:gradients/encoder_1/initial_state_3_tiled_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
|
:gradients/encoder_1/initial_state_3_tiled_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
4gradients/encoder_1/initial_state_3_tiled_grad/rangeRange:gradients/encoder_1/initial_state_3_tiled_grad/range/start3gradients/encoder_1/initial_state_3_tiled_grad/Size:gradients/encoder_1/initial_state_3_tiled_grad/range/delta*
_output_shapes
:*

Tidx0
�
8gradients/encoder_1/initial_state_3_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_5_grad/Exit6gradients/encoder_1/initial_state_3_tiled_grad/Reshape*
Tshape0*J
_output_shapes8
6:4������������������������������������*
T0
�
2gradients/encoder_1/initial_state_3_tiled_grad/SumSum8gradients/encoder_1/initial_state_3_tiled_grad/Reshape_14gradients/encoder_1/initial_state_3_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	�
�
<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5
�
?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
@gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_5^gradients/Add*
_output_shapes
:*
swap_memory( *1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*
T0
�
Hgradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*
T0*
is_constant(
�
?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*(
_output_shapes
:����������
�
6gradients/encoder_1/rnn/while/Select_4_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop*(
_output_shapes
:����������*
T0
�
8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*

stack_name *
_output_shapes
:
�
;gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4
�
<gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_4^gradients/Add*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
_output_shapes
:*
swap_memory( *
T0

�
Dgradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4
�
;gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
_output_shapes	
:�
�
2gradients/encoder_1/rnn/while/Select_4_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_5_grad/Switch:16gradients/encoder_1/rnn/while/Select_4_grad/zeros_like*
T0* 
_output_shapes
:
��
�
4gradients/encoder_1/rnn/while/Select_4_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_4_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_5_grad/Switch:1* 
_output_shapes
:
��*
T0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"�   �   
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
valueB"�   �   *
_output_shapes
:*
dtype0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_accStack*
	elem_type0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
_output_shapes
:*

stack_name 
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc*
T0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnter:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1^gradients/Add*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
_output_shapes
:*
swap_memory( *
T0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc*
is_constant(*
T0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mulMul4gradients/encoder_1/rnn/while/Select_4_grad/Select_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop* 
_output_shapes
:
��*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape* 
_output_shapes
:
��*
Tshape0*
T0
�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_accStack*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
_output_shapes
:*

stack_name 
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2^gradients/Add*
_output_shapes
:*
swap_memory( *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop4gradients/encoder_1/rnn/while/Select_4_grad/Select_1*
T0* 
_output_shapes
:
��
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*
Tshape0* 
_output_shapes
:
��*
T0
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape*
T0* 
_output_shapes
:
��
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1*
T0* 
_output_shapes
:
��
�
gradients/AddN_1AddN4gradients/encoder_1/rnn/while/Select_3_grad/Select_1Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGrad*
N*
T0* 
_output_shapes
:
��*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_3_grad/Select_1
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"�   �   
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
valueB"�   �   *
dtype0*
_output_shapes
:
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/SumSumgradients/AddN_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1* 
_output_shapes
:
��*
Tshape0*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ShapeConst^gradients/Sub*
valueB"�   �   *
_output_shapes
:*
dtype0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1Shapeencoder_1/rnn/while/Identity_4*
_output_shapes
:*
out_type0*
T0
�
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStack*
	elem_type0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1
�
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1
�
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
:*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1
�
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shapeegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*2
_output_shapes 
:���������:���������
�
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mulMulPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop* 
_output_shapes
:
��*
T0
�
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumSumJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape*
T0* 
_output_shapes
:
��*
Tshape0
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_accStack*
	elem_type0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*

stack_name *
_output_shapes
:
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnter;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid^gradients/Add*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
_output_shapes
:*
swap_memory( *
T0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid* 
_output_shapes
:
��
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1MulUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape* 
_output_shapes
:
��*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
Tshape0*(
_output_shapes
:����������*
T0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
valueB"�   �   *
dtype0*
_output_shapes
:
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"�   �   
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_accStack*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
_output_shapes
:*

stack_name 
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnter8encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh^gradients/Add*
_output_shapes
:*
swap_memory( *K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mulMulRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop*
T0* 
_output_shapes
:
��
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape*
Tshape0* 
_output_shapes
:
��*
T0
�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_accStack*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
_output_shapes
:*

stack_name 
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
T0
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1^gradients/Add*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
_output_shapes
:*
swap_memory( *
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
T0
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1* 
_output_shapes
:
��
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1* 
_output_shapes
:
��*
T0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1*
T0* 
_output_shapes
:
��*
Tshape0
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape*
T0* 
_output_shapes
:
��
�
gradients/AddN_2AddN2gradients/encoder_1/rnn/while/Select_3_grad/SelectPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1*
T0*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_3_grad/Select*
N* 
_output_shapes
:
��
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape*
T0* 
_output_shapes
:
��
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1*
T0* 
_output_shapes
:
��
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"�   �   
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
: *
valueB 
�
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumSumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape*
Tshape0* 
_output_shapes
:
��*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1SumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
;gradients/encoder_1/rnn/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_2*
T0* 
_output_shapes
:
��
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim
�
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPush	StackPushXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnterCencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim^gradients/Add*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
_output_shapes
:*
swap_memory( *
T0
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc*
T0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPopStackPopagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop/RefEnter^gradients/Sub*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
_output_shapes
: 
�
Ogradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concatConcatV2Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop*

Tidx0*
T0*
N* 
_output_shapes
:
��
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
_output_shapes	
:�*
data_formatNHWC*
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
��*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMulMatMulOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(* 
_output_shapes
:
��*
transpose_a( *
T0
�
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_accStack*
	elem_type0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*

stack_name *
_output_shapes
:
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
T0*
is_constant(
�
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat^gradients/Add*
T0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
_output_shapes
:*
swap_memory( 
�
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat
�
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1MatMulegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
transpose_b( * 
_output_shapes
:
��*
transpose_a(*
T0
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
_output_shapes	
:�*
valueB�*    
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc*
parallel_iterations *
T0*
_output_shapes	
:�*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:�: 
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*"
_output_shapes
:�:�*
T0
�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/AddAddYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch:1Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:�*
T0
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:�
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_accStack*
	elem_type0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*

stack_name *
_output_shapes
:
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
T0*
is_constant(
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPush	StackPush`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnterIencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis^gradients/Add*
_output_shapes
:*
swap_memory( *\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
T0
�
igradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopStackPopigradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
: *\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/modFloorMod`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
�
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"�   �   
�
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Shape_1Shapeencoder_1/rnn/while/Identity_5*
out_type0*
_output_shapes
:*
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2
�
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
T0*
is_constant(
�
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPush	StackPushcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2^gradients/Add*
T0*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:*
swap_memory( 
�
lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
T0
�
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopStackPoplgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnter^gradients/Sub*
	elem_type0*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2* 
_output_shapes
:
��
�
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeNShapeNcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop*
N*
T0* 
_output_shapes
::*
out_type0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffsetConcatOffsetWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/modZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
�
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/SliceSliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffsetZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN*
Index0*
T0* 
_output_shapes
:
��
�
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice_1SliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMulbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffset:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:����������
�
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
��*    * 
_output_shapes
:
��*
dtype0
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1Enter_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations * 
_output_shapes
:
��*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2Mergeagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIteration*
N*
T0*"
_output_shapes
:
��: 
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/SwitchSwitchagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_output_shapes
:
��:
��*
T0
�
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/AddAddbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIterationNextIteration]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
��*
T0
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3Exit`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
��*
T0
�
gradients/AddN_3AddN4gradients/encoder_1/rnn/while/Select_2_grad/Select_1Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice* 
_output_shapes
:
��*
N*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_2_grad/Select_1*
T0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"�   �   
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
valueB"�   �   *
_output_shapes
:*
dtype0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_accStack*
	elem_type0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
_output_shapes
:*

stack_name 
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnter:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1^gradients/Add*
T0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
_output_shapes
:*
swap_memory( 
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc*
T0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1* 
_output_shapes
:
��
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mulMulgradients/AddN_3Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop*
T0* 
_output_shapes
:
��
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
T0*
is_constant(
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2^gradients/Add*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
_output_shapes
:*
swap_memory( *
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc*
is_constant(*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2* 
_output_shapes
:
��
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopgradients/AddN_3* 
_output_shapes
:
��*
T0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*
Tshape0* 
_output_shapes
:
��*
T0
�
gradients/AddN_4AddN2gradients/encoder_1/rnn/while/Select_4_grad/Select[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice_1* 
_output_shapes
:
��*
N*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_4_grad/Select*
T0
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape*
T0* 
_output_shapes
:
��
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1*
T0* 
_output_shapes
:
��
�
;gradients/encoder_1/rnn/while/Switch_5_grad_1/NextIterationNextIterationgradients/AddN_4* 
_output_shapes
:
��*
T0
�
gradients/AddN_5AddN4gradients/encoder_1/rnn/while/Select_1_grad/Select_1Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGrad*
T0*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_1_grad/Select_1*
N* 
_output_shapes
:
��
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
valueB"�   �   *
dtype0*
_output_shapes
:
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"�   �   
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/SumSumgradients/AddN_5^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape*
T0* 
_output_shapes
:
��*
Tshape0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_5`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*
T0*
Tshape0* 
_output_shapes
:
��
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ShapeConst^gradients/Sub*
valueB"�   �   *
_output_shapes
:*
dtype0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1Shapeencoder_1/rnn/while/Identity_2*
T0*
out_type0*
_output_shapes
:
�
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStack*
	elem_type0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*

stack_name 
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
is_constant(*
T0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
�
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1^gradients/Add*
T0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*
swap_memory( 
�
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
:*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1
�
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shapeegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*2
_output_shapes 
:���������:���������
�
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mulMulPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop* 
_output_shapes
:
��*
T0
�
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumSumJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_accStack*
	elem_type0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
_output_shapes
:*

stack_name 
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc*
T0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnter;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid^gradients/Add*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
_output_shapes
:*
swap_memory( *
T0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1MulUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape* 
_output_shapes
:
��*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*(
_output_shapes
:����������*
Tshape0*
T0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"�   �   
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
valueB"�   �   *
_output_shapes
:*
dtype0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_accStack*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*

stack_name *
_output_shapes
:
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnter8encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc*
is_constant(*
T0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
��*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mulMulRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop* 
_output_shapes
:
��*
T0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape* 
_output_shapes
:
��*
Tshape0*
T0
�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_accStack*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
_output_shapes
:*

stack_name 
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1^gradients/Add*
_output_shapes
:*
swap_memory( *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1* 
_output_shapes
:
��
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1* 
_output_shapes
:
��*
T0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*
T0*
Tshape0* 
_output_shapes
:
��
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape*
T0* 
_output_shapes
:
��
�
gradients/AddN_6AddN2gradients/encoder_1/rnn/while/Select_1_grad/SelectPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1*
N*
T0* 
_output_shapes
:
��*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_1_grad/Select
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape* 
_output_shapes
:
��*
T0
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1*
T0* 
_output_shapes
:
��
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeConst^gradients/Sub*
valueB"�   �   *
_output_shapes
:*
dtype0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
valueB *
_output_shapes
: *
dtype0
�
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumSumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape*
T0* 
_output_shapes
:
��*
Tshape0
�
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1SumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
;gradients/encoder_1/rnn/while/Switch_2_grad_1/NextIterationNextIterationgradients/AddN_6*
T0* 
_output_shapes
:
��
�
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_accStack*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*

stack_name *
_output_shapes
:
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
T0*
is_constant(
�
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPush	StackPushXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnterCencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim^gradients/Add*
T0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
_output_shapes
:*
swap_memory( 
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPopStackPopagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
: *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim
�
Ogradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concatConcatV2Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop*
N*

Tidx0*
T0* 
_output_shapes
:
��
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
_output_shapes	
:�*
data_formatNHWC*
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read* 
_output_shapes
:
��*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMulMatMulOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(* 
_output_shapes
:
��*
transpose_a( *
T0
�
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_accStack*
	elem_type0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat*
_output_shapes
:*

stack_name 
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat*
T0
�
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat^gradients/Add*
_output_shapes
:*
swap_memory( *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat*
T0
�
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
�
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat* 
_output_shapes
:
��
�
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1MatMulegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
transpose_b( * 
_output_shapes
:
��*
transpose_a(*
T0
�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB�*    *
_output_shapes	
:�*
dtype0
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc*
_output_shapes	
:�*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
T0*
_output_shapes
	:�: 
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
:�:�
�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/AddAddYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch:1Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:�*
T0
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:�*
T0
�
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
_output_shapes
: *
dtype0
�
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
T0*
is_constant(
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPush	StackPush`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnterIencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis^gradients/Add*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*
swap_memory( *
T0
�
igradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
is_constant(*
T0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopStackPopigradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnter^gradients/Sub*
	elem_type0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
_output_shapes
: 
�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/modFloorMod`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
�
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"�   �   
�
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Shape_1Shapeencoder_1/rnn/while/Identity_3*
_output_shapes
:*
out_type0*
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *.
_class$
" loc:@encoder_1/rnn/TensorArray_1
�
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
�
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPush	StackPushcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter%encoder_1/rnn/while/TensorArrayReadV3^gradients/Add*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
:*
swap_memory( *
T0
�
lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
�
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopStackPoplgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnter^gradients/Sub*
	elem_type0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1* 
_output_shapes
:
��
�
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeNShapeNcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop* 
_output_shapes
::*
N*
out_type0*
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffsetConcatOffsetWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/modZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
�
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/SliceSliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffsetZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN* 
_output_shapes
:
��*
Index0*
T0
�
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Slice_1SliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMulbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffset:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN:1*(
_output_shapes
:����������*
Index0*
T0
�
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_accConst* 
_output_shapes
:
��*
dtype0*
valueB
��*    
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1Enter_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc* 
_output_shapes
:
��*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2Mergeagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIteration*"
_output_shapes
:
��: *
N*
T0
�
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/SwitchSwitchagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
��:
��
�
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/AddAddbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIterationNextIteration]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
��
�
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3Exit`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
��
�
\gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterencoder_1/rnn/TensorArray_1*
is_constant(*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
�
^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterHencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
�
Vgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3\gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*
source	gradients*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes

::
�
Rgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1W^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: 
�
^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_accStack*
	elem_type0*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*

stack_name *
_output_shapes
:
�
agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnterRefEnter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*
is_constant(*
T0*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
�
bgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPush	StackPushagradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnterencoder_1/rnn/while/Identity^gradients/Add*
_output_shapes
:*
swap_memory( *Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
T0
�
jgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop/RefEnterRefEnter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
�
agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopStackPopjgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop/RefEnter^gradients/Sub*
	elem_type0*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
: 
�
Xgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Vgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/SliceRgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: 
�
gradients/AddN_7AddN2gradients/encoder_1/rnn/while/Select_2_grad/Select[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Slice_1*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_2_grad/Select* 
_output_shapes
:
��*
T0*
N
�
Bgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterBgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
�
Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Jgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
N*
T0*
_output_shapes
: : 
�
Cgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*
T0*
_output_shapes
: : 
�
@gradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddEgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Xgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
�
Jgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration@gradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
�
Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitCgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
_output_shapes
: *
T0
�
;gradients/encoder_1/rnn/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_7* 
_output_shapes
:
��*
T0
�
ygradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3encoder_1/rnn/TensorArray_1Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
source	gradients*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes

::
�
ugradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3z^gradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: *
T0
�
kgradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3ygradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3&encoder_1/rnn/TensorArrayUnstack/rangeugradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
element_shape:*%
_output_shapes
:���*
dtype0
�
4gradients/encoder_1/transpose_grad/InvertPermutationInvertPermutationencoder_1/transpose/perm*
T0*
_output_shapes
:
�
,gradients/encoder_1/transpose_grad/transpose	Transposekgradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV34gradients/encoder_1/transpose_grad/InvertPermutation*
Tperm0*
T0*%
_output_shapes
:���
�
+gradients/encoder/conv1d/Squeeze_grad/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"�      �   �   
�
-gradients/encoder/conv1d/Squeeze_grad/ReshapeReshape,gradients/encoder_1/transpose_grad/transpose+gradients/encoder/conv1d/Squeeze_grad/Shape*)
_output_shapes
:���*
Tshape0*
T0
�
*gradients/encoder/conv1d/Conv2D_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"�      �      
�
8gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/encoder/conv1d/Conv2D_grad/Shapeencoder/conv1d/ExpandDims_1-gradients/encoder/conv1d/Squeeze_grad/Reshape*
use_cudnn_on_gpu(*(
_output_shapes
:��*
data_formatNHWC*
strides
*
T0*
paddingVALID
�
,gradients/encoder/conv1d/Conv2D_grad/Shape_1Const*%
valueB"         �   *
dtype0*
_output_shapes
:
�
9gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/conv1d/ExpandDims,gradients/encoder/conv1d/Conv2D_grad/Shape_1-gradients/encoder/conv1d/Squeeze_grad/Reshape*
paddingVALID*
T0*
data_formatNHWC*
strides
*'
_output_shapes
:�*
use_cudnn_on_gpu(
�
0gradients/encoder/conv1d/ExpandDims_1_grad/ShapeConst*!
valueB"      �   *
dtype0*
_output_shapes
:
�
2gradients/encoder/conv1d/ExpandDims_1_grad/ReshapeReshape9gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropFilter0gradients/encoder/conv1d/ExpandDims_1_grad/Shape*#
_output_shapes
:�*
Tshape0*
T0
�
global_norm/L2LossL2Loss2gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*
_output_shapes
: *E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*
T0
�
global_norm/L2Loss_1L2Loss2gradients/encoder_1/initial_state_0_tiled_grad/Sum*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
: 
�
global_norm/L2Loss_2L2Loss2gradients/encoder_1/initial_state_1_tiled_grad/Sum*
_output_shapes
: *E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
T0
�
global_norm/L2Loss_3L2Loss2gradients/encoder_1/initial_state_2_tiled_grad/Sum*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum*
_output_shapes
: *
T0
�
global_norm/L2Loss_4L2Loss2gradients/encoder_1/initial_state_3_tiled_grad/Sum*
T0*
_output_shapes
: *E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum
�
global_norm/L2Loss_5L2Lossagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *
T0
�
global_norm/L2Loss_6L2LossXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: *
T0
�
global_norm/L2Loss_7L2Lossagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: 
�
global_norm/L2Loss_8L2LossXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: *k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
global_norm/L2Loss_9L2Loss:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
T0*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
_output_shapes
: 
�
global_norm/L2Loss_10L2Loss6gradients/output_projection/xw_plus_b_grad/BiasAddGrad*I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
_output_shapes
: *
T0
�
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
global_norm/Const_1Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
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
 *  �?*
dtype0*
_output_shapes
: 

clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xglobal_norm/global_norm*
T0*
_output_shapes
: 
^
clip_by_global_norm/ConstConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
d
clip_by_global_norm/truediv_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
�
clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
_output_shapes
: *
T0
�
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
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
_output_shapes
: *
T0
�
clip_by_global_norm/mul_1Mul2gradients/encoder/conv1d/ExpandDims_1_grad/Reshapeclip_by_global_norm/mul*
T0*#
_output_shapes
:�*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape
�
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*
T0*#
_output_shapes
:�*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape
�
clip_by_global_norm/mul_2Mul2gradients/encoder_1/initial_state_0_tiled_grad/Sumclip_by_global_norm/mul*
T0*
_output_shapes
:	�*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum
�
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
:	�
�
clip_by_global_norm/mul_3Mul2gradients/encoder_1/initial_state_1_tiled_grad/Sumclip_by_global_norm/mul*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
_output_shapes
:	�
�
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
_output_shapes
:	�*
T0
�
clip_by_global_norm/mul_4Mul2gradients/encoder_1/initial_state_2_tiled_grad/Sumclip_by_global_norm/mul*
_output_shapes
:	�*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum*
T0
�
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
_output_shapes
:	�*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum*
T0
�
clip_by_global_norm/mul_5Mul2gradients/encoder_1/initial_state_3_tiled_grad/Sumclip_by_global_norm/mul*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
_output_shapes
:	�*
T0
�
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*
_output_shapes
:	�*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
T0
�
clip_by_global_norm/mul_6Mulagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
��*
T0
�
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*
T0*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
��
�
clip_by_global_norm/mul_7MulXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0*
_output_shapes	
:�*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3
�
*clip_by_global_norm/clip_by_global_norm/_6Identityclip_by_global_norm/mul_7*
T0*
_output_shapes	
:�*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3
�
clip_by_global_norm/mul_8Mulagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
��*
T0
�
*clip_by_global_norm/clip_by_global_norm/_7Identityclip_by_global_norm/mul_8*
T0* 
_output_shapes
:
��*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3
�
clip_by_global_norm/mul_9MulXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:�
�
*clip_by_global_norm/clip_by_global_norm/_8Identityclip_by_global_norm/mul_9*
T0*
_output_shapes	
:�*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3
�
clip_by_global_norm/mul_10Mul:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1clip_by_global_norm/mul*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
_output_shapes
:	�*
T0
�
*clip_by_global_norm/clip_by_global_norm/_9Identityclip_by_global_norm/mul_10*
T0*
_output_shapes
:	�*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1
�
clip_by_global_norm/mul_11Mul6gradients/output_projection/xw_plus_b_grad/BiasAddGradclip_by_global_norm/mul*I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
_output_shapes
:*
T0
�
+clip_by_global_norm/clip_by_global_norm/_10Identityclip_by_global_norm/mul_11*
T0*
_output_shapes
:*I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad
p
grad_norms/grad_norms/tagsConst*
_output_shapes
: *
dtype0*&
valueB Bgrad_norms/grad_norms
|
grad_norms/grad_normsScalarSummarygrad_norms/grad_norms/tagsglobal_norm/global_norm*
_output_shapes
: *
T0
~
beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*
_class
loc:@input_embed
�
beta1_power
VariableV2*
	container *
dtype0*
_class
loc:@input_embed*
_output_shapes
: *
shape: *
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_class
loc:@input_embed*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
j
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@input_embed*
_output_shapes
: 
~
beta2_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *w�?*
_class
loc:@input_embed
�
beta2_power
VariableV2*
	container *
dtype0*
_class
loc:@input_embed*
shared_name *
_output_shapes
: *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_class
loc:@input_embed*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
j
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@input_embed*
_output_shapes
: 
d
zerosConst*
dtype0*#
_output_shapes
:�*"
valueB�*    
�
input_embed/Adam
VariableV2*
	container *
dtype0*
_class
loc:@input_embed*
shared_name *#
_output_shapes
:�*
shape:�
�
input_embed/Adam/AssignAssigninput_embed/Adamzeros*#
_output_shapes
:�*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
�
input_embed/Adam/readIdentityinput_embed/Adam*
T0*#
_output_shapes
:�*
_class
loc:@input_embed
f
zeros_1Const*
dtype0*#
_output_shapes
:�*"
valueB�*    
�
input_embed/Adam_1
VariableV2*
shared_name *
_class
loc:@input_embed*
	container *
shape:�*
dtype0*#
_output_shapes
:�
�
input_embed/Adam_1/AssignAssigninput_embed/Adam_1zeros_1*
_class
loc:@input_embed*#
_output_shapes
:�*
T0*
validate_shape(*
use_locking(
�
input_embed/Adam_1/readIdentityinput_embed/Adam_1*
T0*
_class
loc:@input_embed*#
_output_shapes
:�
^
zeros_2Const*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
encoder/initial_state_0/Adam
VariableV2*
_output_shapes
:	�*
dtype0*
shape:	�*
	container **
_class 
loc:@encoder/initial_state_0*
shared_name 
�
#encoder/initial_state_0/Adam/AssignAssignencoder/initial_state_0/Adamzeros_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_0
�
!encoder/initial_state_0/Adam/readIdentityencoder/initial_state_0/Adam*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_0
^
zeros_3Const*
_output_shapes
:	�*
dtype0*
valueB	�*    
�
encoder/initial_state_0/Adam_1
VariableV2*
shared_name **
_class 
loc:@encoder/initial_state_0*
	container *
shape:	�*
dtype0*
_output_shapes
:	�
�
%encoder/initial_state_0/Adam_1/AssignAssignencoder/initial_state_0/Adam_1zeros_3*
_output_shapes
:	�*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(
�
#encoder/initial_state_0/Adam_1/readIdentityencoder/initial_state_0/Adam_1*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_0
^
zeros_4Const*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
encoder/initial_state_1/Adam
VariableV2*
shape:	�*
_output_shapes
:	�*
shared_name **
_class 
loc:@encoder/initial_state_1*
dtype0*
	container 
�
#encoder/initial_state_1/Adam/AssignAssignencoder/initial_state_1/Adamzeros_4**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
!encoder/initial_state_1/Adam/readIdentityencoder/initial_state_1/Adam*
T0**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	�
^
zeros_5Const*
valueB	�*    *
_output_shapes
:	�*
dtype0
�
encoder/initial_state_1/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
shape:	�*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_1
�
%encoder/initial_state_1/Adam_1/AssignAssignencoder/initial_state_1/Adam_1zeros_5*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	�
�
#encoder/initial_state_1/Adam_1/readIdentityencoder/initial_state_1/Adam_1*
T0**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	�
^
zeros_6Const*
valueB	�*    *
_output_shapes
:	�*
dtype0
�
encoder/initial_state_2/Adam
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	�*
shape:	�*
shared_name 
�
#encoder/initial_state_2/Adam/AssignAssignencoder/initial_state_2/Adamzeros_6**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
!encoder/initial_state_2/Adam/readIdentityencoder/initial_state_2/Adam*
T0**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	�
^
zeros_7Const*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
encoder/initial_state_2/Adam_1
VariableV2*
_output_shapes
:	�*
dtype0*
shape:	�*
	container **
_class 
loc:@encoder/initial_state_2*
shared_name 
�
%encoder/initial_state_2/Adam_1/AssignAssignencoder/initial_state_2/Adam_1zeros_7*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_2
�
#encoder/initial_state_2/Adam_1/readIdentityencoder/initial_state_2/Adam_1**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	�*
T0
^
zeros_8Const*
dtype0*
_output_shapes
:	�*
valueB	�*    
�
encoder/initial_state_3/Adam
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_3*
shared_name *
_output_shapes
:	�*
shape:	�
�
#encoder/initial_state_3/Adam/AssignAssignencoder/initial_state_3/Adamzeros_8*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	�
�
!encoder/initial_state_3/Adam/readIdentityencoder/initial_state_3/Adam*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_3*
T0
^
zeros_9Const*
valueB	�*    *
_output_shapes
:	�*
dtype0
�
encoder/initial_state_3/Adam_1
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_3*
shared_name *
_output_shapes
:	�*
shape:	�
�
%encoder/initial_state_3/Adam_1/AssignAssignencoder/initial_state_3/Adam_1zeros_9*
_output_shapes
:	�*
validate_shape(**
_class 
loc:@encoder/initial_state_3*
T0*
use_locking(
�
#encoder/initial_state_3/Adam_1/readIdentityencoder/initial_state_3/Adam_1**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	�*
T0
a
zeros_10Const*
valueB
��*    *
dtype0* 
_output_shapes
:
��
�
8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam
VariableV2*
shared_name *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/AssignAssign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamzeros_10*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
validate_shape(* 
_output_shapes
:
��
�
=encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/readIdentity8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
��*
T0
a
zeros_11Const*
dtype0* 
_output_shapes
:
��*
valueB
��*    
�
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1
VariableV2*
	container *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
shared_name * 
_output_shapes
:
��*
shape:
��
�
Aencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/AssignAssign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1zeros_11* 
_output_shapes
:
��*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0*
use_locking(
�
?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/readIdentity:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0
W
zeros_12Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam
VariableV2*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adamzeros_12*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:�*
T0*
validate_shape(*
use_locking(
�
<encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/readIdentity7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
T0
W
zeros_13Const*
_output_shapes	
:�*
dtype0*
valueB�*    
�
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1
VariableV2*
_output_shapes	
:�*
dtype0*
shape:�*
	container *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
shared_name 
�
@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1zeros_13*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
�
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/readIdentity9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*
T0*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
a
zeros_14Const*
valueB
��*    *
dtype0* 
_output_shapes
:
��
�
8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam
VariableV2*
shape:
��* 
_output_shapes
:
��*
shared_name *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
	container 
�
?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/AssignAssign8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adamzeros_14* 
_output_shapes
:
��*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0*
use_locking(
�
=encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/readIdentity8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam* 
_output_shapes
:
��*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0
a
zeros_15Const*
dtype0* 
_output_shapes
:
��*
valueB
��*    
�
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1
VariableV2* 
_output_shapes
:
��*
dtype0*
shape:
��*
	container *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
shared_name 
�
Aencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/AssignAssign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1zeros_15* 
_output_shapes
:
��*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0*
use_locking(
�
?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/readIdentity:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
��*
T0
W
zeros_16Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam
VariableV2*
shared_name *
shape:�*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
dtype0*
	container 
�
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adamzeros_16*
_output_shapes	
:�*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
use_locking(
�
<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/readIdentity7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*
T0*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
W
zeros_17Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1
VariableV2*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1zeros_17*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:�*
T0*
validate_shape(*
use_locking(
�
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/readIdentity9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0
_
zeros_18Const*
dtype0*
_output_shapes
:	�*
valueB	�*    
�
output_projection/W/Adam
VariableV2*
	container *
dtype0*&
_class
loc:@output_projection/W*
shared_name *
_output_shapes
:	�*
shape:	�
�
output_projection/W/Adam/AssignAssignoutput_projection/W/Adamzeros_18*&
_class
loc:@output_projection/W*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
output_projection/W/Adam/readIdentityoutput_projection/W/Adam*
T0*
_output_shapes
:	�*&
_class
loc:@output_projection/W
_
zeros_19Const*
_output_shapes
:	�*
dtype0*
valueB	�*    
�
output_projection/W/Adam_1
VariableV2*
shared_name *&
_class
loc:@output_projection/W*
	container *
shape:	�*
dtype0*
_output_shapes
:	�
�
!output_projection/W/Adam_1/AssignAssignoutput_projection/W/Adam_1zeros_19*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�*&
_class
loc:@output_projection/W
�
output_projection/W/Adam_1/readIdentityoutput_projection/W/Adam_1*
T0*&
_class
loc:@output_projection/W*
_output_shapes
:	�
U
zeros_20Const*
valueB*    *
_output_shapes
:*
dtype0
�
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
�
output_projection/b/Adam/AssignAssignoutput_projection/b/Adamzeros_20*
use_locking(*
T0*&
_class
loc:@output_projection/b*
validate_shape(*
_output_shapes
:
�
output_projection/b/Adam/readIdentityoutput_projection/b/Adam*
_output_shapes
:*&
_class
loc:@output_projection/b*
T0
U
zeros_21Const*
_output_shapes
:*
dtype0*
valueB*    
�
output_projection/b/Adam_1
VariableV2*
_output_shapes
:*
dtype0*
shape:*
	container *&
_class
loc:@output_projection/b*
shared_name 
�
!output_projection/b/Adam_1/AssignAssignoutput_projection/b/Adam_1zeros_21*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(
�
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
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *w�+2
�
!Adam/update_input_embed/ApplyAdam	ApplyAdaminput_embedinput_embed/Adaminput_embed/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
use_locking( *
T0*#
_output_shapes
:�*
_class
loc:@input_embed
�
-Adam/update_encoder/initial_state_0/ApplyAdam	ApplyAdamencoder/initial_state_0encoder/initial_state_0/Adamencoder/initial_state_0/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	�*
T0*
use_locking( 
�
-Adam/update_encoder/initial_state_1/ApplyAdam	ApplyAdamencoder/initial_state_1encoder/initial_state_1/Adamencoder/initial_state_1/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
use_locking( *
T0**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	�
�
-Adam/update_encoder/initial_state_2/ApplyAdam	ApplyAdamencoder/initial_state_2encoder/initial_state_2/Adamencoder/initial_state_2/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
use_locking( *
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_2
�
-Adam/update_encoder/initial_state_3/ApplyAdam	ApplyAdamencoder/initial_state_3encoder/initial_state_3/Adamencoder/initial_state_3/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_4*
use_locking( *
T0**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	�
�
IAdam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdam	ApplyAdam3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_5*
use_locking( *
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
��
�
HAdam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdam	ApplyAdam2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_6*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:�*
T0*
use_locking( 
�
IAdam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdam	ApplyAdam3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_7*
use_locking( *
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
��
�
HAdam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam	ApplyAdam2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_8*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
use_locking( 
�
)Adam/update_output_projection/W/ApplyAdam	ApplyAdamoutput_projection/Woutput_projection/W/Adamoutput_projection/W/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_9*
use_locking( *
T0*
_output_shapes
:	�*&
_class
loc:@output_projection/W
�
)Adam/update_output_projection/b/ApplyAdam	ApplyAdamoutput_projection/boutput_projection/b/Adamoutput_projection/b/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_10*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0*
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@input_embed
�
Adam/AssignAssignbeta1_powerAdam/mul*
_class
loc:@input_embed*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam*
_output_shapes
: *
_class
loc:@input_embed*
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_class
loc:@input_embed*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
�
Adam/updateNoOp"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam^Adam/Assign^Adam/Assign_1
w

Adam/valueConst^Adam/update*
value	B :*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
x
Adam	AssignAddVariable
Adam/value*
use_locking( *
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Const_1Const*
_output_shapes	
:�*
dtype0	*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
b
ArgMaxArgMaxcond/Merge_1ArgMax/dimension*
_output_shapes	
:�*
T0*

Tidx0
E
EqualEqualArgMaxConst_1*
T0	*
_output_shapes	
:�
L

LogicalAnd
LogicalAndaccuracy/EqualEqual*
_output_shapes	
:�
Z
count_nonzero/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
n
count_nonzero/NotEqualNotEqual
LogicalAndcount_nonzero/NotEqual/y*
_output_shapes	
:�*
T0

j
count_nonzero/ToInt64Castcount_nonzero/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

]
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
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
:�
\
count_nonzero_1/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

o
count_nonzero_1/NotEqualNotEqualEqual_1count_nonzero_1/NotEqual/y*
_output_shapes	
:�*
T0

n
count_nonzero_1/ToInt64Castcount_nonzero_1/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

_
count_nonzero_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
count_nonzero_1/SumSumcount_nonzero_1/ToInt64count_nonzero_1/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
T
ArgMax_1/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
f
ArgMax_1ArgMaxcond/Merge_1ArgMax_1/dimension*

Tidx0*
T0*
_output_shapes	
:�
I
Equal_2EqualArgMax_1Const_1*
T0	*
_output_shapes	
:�
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
:�
n
count_nonzero_2/ToInt64Castcount_nonzero_2/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

_
count_nonzero_2/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
count_nonzero_2/SumSumcount_nonzero_2/ToInt64count_nonzero_2/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
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
cond_1/switch_fIdentitycond_1/Switch*
_output_shapes
: *
T0

D
cond_1/pred_idIdentityGreater*
T0
*
_output_shapes
: 
�
cond_1/truediv/Cast/SwitchSwitchcount_nonzero/Sumcond_1/pred_id*$
_class
loc:@count_nonzero/Sum*
_output_shapes
: : *
T0	
i
cond_1/truediv/CastCastcond_1/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
�
cond_1/truediv/Cast_1/SwitchSwitchcount_nonzero_1/Sumcond_1/pred_id*
T0	*&
_class
loc:@count_nonzero_1/Sum*
_output_shapes
: : 
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
cond_1/ConstConst^cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB 2        
_
cond_1/MergeMergecond_1/Constcond_1/truediv*
T0*
N*
_output_shapes
: : 
\
precision_0/tagsConst*
valueB Bprecision_0*
dtype0*
_output_shapes
: 
]
precision_0ScalarSummaryprecision_0/tagscond_1/Merge*
T0*
_output_shapes
: 
M
Greater_1/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
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

�
cond_2/truediv/Cast/SwitchSwitchcount_nonzero/Sumcond_2/pred_id*$
_class
loc:@count_nonzero/Sum*
_output_shapes
: : *
T0	
i
cond_2/truediv/CastCastcond_2/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
�
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
recall_0/tagsConst*
dtype0*
_output_shapes
: *
valueB Brecall_0
W
recall_0ScalarSummaryrecall_0/tagscond_2/Merge*
T0*
_output_shapes
: 
�
Const_2Const*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                *
_output_shapes	
:�*
dtype0	
T
ArgMax_2/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
f
ArgMax_2ArgMaxcond/Merge_1ArgMax_2/dimension*
_output_shapes	
:�*
T0*

Tidx0
I
Equal_3EqualArgMax_2Const_2*
T0	*
_output_shapes	
:�
P
LogicalAnd_1
LogicalAndaccuracy/EqualEqual_3*
_output_shapes	
:�
\
count_nonzero_3/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
t
count_nonzero_3/NotEqualNotEqualLogicalAnd_1count_nonzero_3/NotEqual/y*
T0
*
_output_shapes	
:�
n
count_nonzero_3/ToInt64Castcount_nonzero_3/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

_
count_nonzero_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
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
:�
\
count_nonzero_4/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
o
count_nonzero_4/NotEqualNotEqualEqual_4count_nonzero_4/NotEqual/y*
_output_shapes	
:�*
T0

n
count_nonzero_4/ToInt64Castcount_nonzero_4/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
_
count_nonzero_4/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
count_nonzero_4/SumSumcount_nonzero_4/ToInt64count_nonzero_4/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
T
ArgMax_3/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
f
ArgMax_3ArgMaxcond/Merge_1ArgMax_3/dimension*
_output_shapes	
:�*
T0*

Tidx0
I
Equal_5EqualArgMax_3Const_2*
_output_shapes	
:�*
T0	
\
count_nonzero_5/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

o
count_nonzero_5/NotEqualNotEqualEqual_5count_nonzero_5/NotEqual/y*
T0
*
_output_shapes	
:�
n
count_nonzero_5/ToInt64Castcount_nonzero_5/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
_
count_nonzero_5/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
count_nonzero_5/SumSumcount_nonzero_5/ToInt64count_nonzero_5/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
M
Greater_2/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
W
	Greater_2Greatercount_nonzero_4/SumGreater_2/y*
T0	*
_output_shapes
: 
P
cond_3/SwitchSwitch	Greater_2	Greater_2*
_output_shapes
: : *
T0

M
cond_3/switch_tIdentitycond_3/Switch:1*
T0
*
_output_shapes
: 
K
cond_3/switch_fIdentitycond_3/Switch*
T0
*
_output_shapes
: 
F
cond_3/pred_idIdentity	Greater_2*
T0
*
_output_shapes
: 
�
cond_3/truediv/Cast/SwitchSwitchcount_nonzero_3/Sumcond_3/pred_id*
_output_shapes
: : *&
_class
loc:@count_nonzero_3/Sum*
T0	
i
cond_3/truediv/CastCastcond_3/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
�
cond_3/truediv/Cast_1/SwitchSwitchcount_nonzero_4/Sumcond_3/pred_id*
T0	*
_output_shapes
: : *&
_class
loc:@count_nonzero_4/Sum
m
cond_3/truediv/Cast_1Castcond_3/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
f
cond_3/truedivRealDivcond_3/truediv/Castcond_3/truediv/Cast_1*
_output_shapes
: *
T0
g
cond_3/ConstConst^cond_3/switch_f*
valueB 2        *
dtype0*
_output_shapes
: 
_
cond_3/MergeMergecond_3/Constcond_3/truediv*
_output_shapes
: : *
T0*
N
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
cond_4/switch_tIdentitycond_4/Switch:1*
T0
*
_output_shapes
: 
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
�
cond_4/truediv/Cast/SwitchSwitchcount_nonzero_3/Sumcond_4/pred_id*
_output_shapes
: : *&
_class
loc:@count_nonzero_3/Sum*
T0	
i
cond_4/truediv/CastCastcond_4/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
�
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
cond_4/MergeMergecond_4/Constcond_4/truediv*
_output_shapes
: : *
T0*
N
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
�
Const_3Const*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                *
dtype0	*
_output_shapes	
:�
T
ArgMax_4/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
f
ArgMax_4ArgMaxcond/Merge_1ArgMax_4/dimension*

Tidx0*
T0*
_output_shapes	
:�
I
Equal_6EqualArgMax_4Const_3*
T0	*
_output_shapes	
:�
P
LogicalAnd_2
LogicalAndaccuracy/EqualEqual_6*
_output_shapes	
:�
\
count_nonzero_6/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
t
count_nonzero_6/NotEqualNotEqualLogicalAnd_2count_nonzero_6/NotEqual/y*
_output_shapes	
:�*
T0

n
count_nonzero_6/ToInt64Castcount_nonzero_6/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

_
count_nonzero_6/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
count_nonzero_6/SumSumcount_nonzero_6/ToInt64count_nonzero_6/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
Y
Equal_7Equaloutput_projection/ArgMaxConst_3*
_output_shapes	
:�*
T0	
\
count_nonzero_7/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

o
count_nonzero_7/NotEqualNotEqualEqual_7count_nonzero_7/NotEqual/y*
_output_shapes	
:�*
T0

n
count_nonzero_7/ToInt64Castcount_nonzero_7/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
_
count_nonzero_7/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
count_nonzero_7/SumSumcount_nonzero_7/ToInt64count_nonzero_7/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
T
ArgMax_5/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
f
ArgMax_5ArgMaxcond/Merge_1ArgMax_5/dimension*

Tidx0*
T0*
_output_shapes	
:�
I
Equal_8EqualArgMax_5Const_3*
T0	*
_output_shapes	
:�
\
count_nonzero_8/NotEqual/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
o
count_nonzero_8/NotEqualNotEqualEqual_8count_nonzero_8/NotEqual/y*
T0
*
_output_shapes	
:�
n
count_nonzero_8/ToInt64Castcount_nonzero_8/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

_
count_nonzero_8/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
count_nonzero_8/SumSumcount_nonzero_8/ToInt64count_nonzero_8/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
M
Greater_4/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
W
	Greater_4Greatercount_nonzero_7/SumGreater_4/y*
_output_shapes
: *
T0	
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
cond_5/switch_fIdentitycond_5/Switch*
T0
*
_output_shapes
: 
F
cond_5/pred_idIdentity	Greater_4*
_output_shapes
: *
T0

�
cond_5/truediv/Cast/SwitchSwitchcount_nonzero_6/Sumcond_5/pred_id*
_output_shapes
: : *&
_class
loc:@count_nonzero_6/Sum*
T0	
i
cond_5/truediv/CastCastcond_5/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
�
cond_5/truediv/Cast_1/SwitchSwitchcount_nonzero_7/Sumcond_5/pred_id*
T0	*
_output_shapes
: : *&
_class
loc:@count_nonzero_7/Sum
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
valueB Bprecision_2*
_output_shapes
: *
dtype0
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
	Greater_5Greatercount_nonzero_8/SumGreater_5/y*
_output_shapes
: *
T0	
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

�
cond_6/truediv/Cast/SwitchSwitchcount_nonzero_6/Sumcond_6/pred_id*&
_class
loc:@count_nonzero_6/Sum*
_output_shapes
: : *
T0	
i
cond_6/truediv/CastCastcond_6/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
�
cond_6/truediv/Cast_1/SwitchSwitchcount_nonzero_8/Sumcond_6/pred_id*
T0	*&
_class
loc:@count_nonzero_8/Sum*
_output_shapes
: : 
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
cond_6/ConstConst^cond_6/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
_
cond_6/MergeMergecond_6/Constcond_6/truediv*
_output_shapes
: : *
N*
T0
V
recall_2/tagsConst*
valueB Brecall_2*
dtype0*
_output_shapes
: 
W
recall_2ScalarSummaryrecall_2/tagscond_6/Merge*
_output_shapes
: *
T0
�
Const_4Const*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                *
dtype0	*
_output_shapes	
:�
T
ArgMax_6/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
f
ArgMax_6ArgMaxcond/Merge_1ArgMax_6/dimension*

Tidx0*
T0*
_output_shapes	
:�
I
Equal_9EqualArgMax_6Const_4*
_output_shapes	
:�*
T0	
P
LogicalAnd_3
LogicalAndaccuracy/EqualEqual_9*
_output_shapes	
:�
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
:�
n
count_nonzero_9/ToInt64Castcount_nonzero_9/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

_
count_nonzero_9/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
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
:�
]
count_nonzero_10/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
r
count_nonzero_10/NotEqualNotEqualEqual_10count_nonzero_10/NotEqual/y*
_output_shapes	
:�*
T0

p
count_nonzero_10/ToInt64Castcount_nonzero_10/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

`
count_nonzero_10/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
count_nonzero_10/SumSumcount_nonzero_10/ToInt64count_nonzero_10/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
T
ArgMax_7/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
f
ArgMax_7ArgMaxcond/Merge_1ArgMax_7/dimension*
_output_shapes	
:�*
T0*

Tidx0
J
Equal_11EqualArgMax_7Const_4*
T0	*
_output_shapes	
:�
]
count_nonzero_11/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

r
count_nonzero_11/NotEqualNotEqualEqual_11count_nonzero_11/NotEqual/y*
T0
*
_output_shapes	
:�
p
count_nonzero_11/ToInt64Castcount_nonzero_11/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

`
count_nonzero_11/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
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
	Greater_6Greatercount_nonzero_10/SumGreater_6/y*
T0	*
_output_shapes
: 
P
cond_7/SwitchSwitch	Greater_6	Greater_6*
_output_shapes
: : *
T0

M
cond_7/switch_tIdentitycond_7/Switch:1*
_output_shapes
: *
T0

K
cond_7/switch_fIdentitycond_7/Switch*
_output_shapes
: *
T0

F
cond_7/pred_idIdentity	Greater_6*
T0
*
_output_shapes
: 
�
cond_7/truediv/Cast/SwitchSwitchcount_nonzero_9/Sumcond_7/pred_id*
T0	*
_output_shapes
: : *&
_class
loc:@count_nonzero_9/Sum
i
cond_7/truediv/CastCastcond_7/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
�
cond_7/truediv/Cast_1/SwitchSwitchcount_nonzero_10/Sumcond_7/pred_id*
T0	*'
_class
loc:@count_nonzero_10/Sum*
_output_shapes
: : 
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
cond_7/ConstConst^cond_7/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
_
cond_7/MergeMergecond_7/Constcond_7/truediv*
N*
T0*
_output_shapes
: : 
\
precision_3/tagsConst*
dtype0*
_output_shapes
: *
valueB Bprecision_3
]
precision_3ScalarSummaryprecision_3/tagscond_7/Merge*
_output_shapes
: *
T0
M
Greater_7/yConst*
_output_shapes
: *
dtype0	*
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
cond_8/switch_tIdentitycond_8/Switch:1*
_output_shapes
: *
T0

K
cond_8/switch_fIdentitycond_8/Switch*
_output_shapes
: *
T0

F
cond_8/pred_idIdentity	Greater_7*
T0
*
_output_shapes
: 
�
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
�
cond_8/truediv/Cast_1/SwitchSwitchcount_nonzero_11/Sumcond_8/pred_id*
T0	*'
_class
loc:@count_nonzero_11/Sum*
_output_shapes
: : 
m
cond_8/truediv/Cast_1Castcond_8/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
f
cond_8/truedivRealDivcond_8/truediv/Castcond_8/truediv/Cast_1*
_output_shapes
: *
T0
g
cond_8/ConstConst^cond_8/switch_f*
valueB 2        *
_output_shapes
: *
dtype0
_
cond_8/MergeMergecond_8/Constcond_8/truediv*
_output_shapes
: : *
N*
T0
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
�
Const_5Const*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                *
_output_shapes	
:�*
dtype0	
T
ArgMax_8/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
f
ArgMax_8ArgMaxcond/Merge_1ArgMax_8/dimension*
_output_shapes	
:�*
T0*

Tidx0
J
Equal_12EqualArgMax_8Const_5*
_output_shapes	
:�*
T0	
Q
LogicalAnd_4
LogicalAndaccuracy/EqualEqual_12*
_output_shapes	
:�
]
count_nonzero_12/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
v
count_nonzero_12/NotEqualNotEqualLogicalAnd_4count_nonzero_12/NotEqual/y*
_output_shapes	
:�*
T0

p
count_nonzero_12/ToInt64Castcount_nonzero_12/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

`
count_nonzero_12/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
count_nonzero_12/SumSumcount_nonzero_12/ToInt64count_nonzero_12/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
Z
Equal_13Equaloutput_projection/ArgMaxConst_5*
_output_shapes	
:�*
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
:�
p
count_nonzero_13/ToInt64Castcount_nonzero_13/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

`
count_nonzero_13/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
count_nonzero_13/SumSumcount_nonzero_13/ToInt64count_nonzero_13/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
T
ArgMax_9/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
f
ArgMax_9ArgMaxcond/Merge_1ArgMax_9/dimension*
_output_shapes	
:�*
T0*

Tidx0
J
Equal_14EqualArgMax_9Const_5*
T0	*
_output_shapes	
:�
]
count_nonzero_14/NotEqual/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
r
count_nonzero_14/NotEqualNotEqualEqual_14count_nonzero_14/NotEqual/y*
_output_shapes	
:�*
T0

p
count_nonzero_14/ToInt64Castcount_nonzero_14/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

`
count_nonzero_14/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
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
cond_9/SwitchSwitch	Greater_8	Greater_8*
T0
*
_output_shapes
: : 
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
cond_9/pred_idIdentity	Greater_8*
_output_shapes
: *
T0

�
cond_9/truediv/Cast/SwitchSwitchcount_nonzero_12/Sumcond_9/pred_id*
T0	*'
_class
loc:@count_nonzero_12/Sum*
_output_shapes
: : 
i
cond_9/truediv/CastCastcond_9/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
�
cond_9/truediv/Cast_1/SwitchSwitchcount_nonzero_13/Sumcond_9/pred_id*
T0	*'
_class
loc:@count_nonzero_13/Sum*
_output_shapes
: : 
m
cond_9/truediv/Cast_1Castcond_9/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
f
cond_9/truedivRealDivcond_9/truediv/Castcond_9/truediv/Cast_1*
_output_shapes
: *
T0
g
cond_9/ConstConst^cond_9/switch_f*
valueB 2        *
dtype0*
_output_shapes
: 
_
cond_9/MergeMergecond_9/Constcond_9/truediv*
_output_shapes
: : *
T0*
N
\
precision_4/tagsConst*
valueB Bprecision_4*
_output_shapes
: *
dtype0
]
precision_4ScalarSummaryprecision_4/tagscond_9/Merge*
T0*
_output_shapes
: 
M
Greater_9/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
X
	Greater_9Greatercount_nonzero_14/SumGreater_9/y*
T0	*
_output_shapes
: 
Q
cond_10/SwitchSwitch	Greater_9	Greater_9*
_output_shapes
: : *
T0

O
cond_10/switch_tIdentitycond_10/Switch:1*
T0
*
_output_shapes
: 
M
cond_10/switch_fIdentitycond_10/Switch*
_output_shapes
: *
T0

G
cond_10/pred_idIdentity	Greater_9*
T0
*
_output_shapes
: 
�
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
�
cond_10/truediv/Cast_1/SwitchSwitchcount_nonzero_14/Sumcond_10/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_14/Sum*
T0	
o
cond_10/truediv/Cast_1Castcond_10/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
i
cond_10/truedivRealDivcond_10/truediv/Castcond_10/truediv/Cast_1*
_output_shapes
: *
T0
i
cond_10/ConstConst^cond_10/switch_f*
_output_shapes
: *
dtype0*
valueB 2        
b
cond_10/MergeMergecond_10/Constcond_10/truediv*
T0*
N*
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
�
Const_6Const*
_output_shapes	
:�*
dtype0	*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
U
ArgMax_10/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
h
	ArgMax_10ArgMaxcond/Merge_1ArgMax_10/dimension*
_output_shapes	
:�*
T0*

Tidx0
K
Equal_15Equal	ArgMax_10Const_6*
_output_shapes	
:�*
T0	
Q
LogicalAnd_5
LogicalAndaccuracy/EqualEqual_15*
_output_shapes	
:�
]
count_nonzero_15/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

v
count_nonzero_15/NotEqualNotEqualLogicalAnd_5count_nonzero_15/NotEqual/y*
T0
*
_output_shapes	
:�
p
count_nonzero_15/ToInt64Castcount_nonzero_15/NotEqual*
_output_shapes	
:�*

DstT0	*

SrcT0

`
count_nonzero_15/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
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
:�
]
count_nonzero_16/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

r
count_nonzero_16/NotEqualNotEqualEqual_16count_nonzero_16/NotEqual/y*
T0
*
_output_shapes	
:�
p
count_nonzero_16/ToInt64Castcount_nonzero_16/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
`
count_nonzero_16/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
count_nonzero_16/SumSumcount_nonzero_16/ToInt64count_nonzero_16/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
U
ArgMax_11/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
h
	ArgMax_11ArgMaxcond/Merge_1ArgMax_11/dimension*

Tidx0*
T0*
_output_shapes	
:�
K
Equal_17Equal	ArgMax_11Const_6*
T0	*
_output_shapes	
:�
]
count_nonzero_17/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
r
count_nonzero_17/NotEqualNotEqualEqual_17count_nonzero_17/NotEqual/y*
T0
*
_output_shapes	
:�
p
count_nonzero_17/ToInt64Castcount_nonzero_17/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
`
count_nonzero_17/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
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
Greater_10*
T0
*
_output_shapes
: : 
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
Greater_10*
T0
*
_output_shapes
: 
�
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
�
cond_11/truediv/Cast_1/SwitchSwitchcount_nonzero_16/Sumcond_11/pred_id*
T0	*
_output_shapes
: : *'
_class
loc:@count_nonzero_16/Sum
o
cond_11/truediv/Cast_1Castcond_11/truediv/Cast_1/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
i
cond_11/truedivRealDivcond_11/truediv/Castcond_11/truediv/Cast_1*
_output_shapes
: *
T0
i
cond_11/ConstConst^cond_11/switch_f*
dtype0*
_output_shapes
: *
valueB 2        
b
cond_11/MergeMergecond_11/Constcond_11/truediv*
_output_shapes
: : *
N*
T0
\
precision_5/tagsConst*
dtype0*
_output_shapes
: *
valueB Bprecision_5
^
precision_5ScalarSummaryprecision_5/tagscond_11/Merge*
T0*
_output_shapes
: 
N
Greater_11/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
Z

Greater_11Greatercount_nonzero_17/SumGreater_11/y*
_output_shapes
: *
T0	
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
cond_12/switch_fIdentitycond_12/Switch*
_output_shapes
: *
T0

H
cond_12/pred_idIdentity
Greater_11*
T0
*
_output_shapes
: 
�
cond_12/truediv/Cast/SwitchSwitchcount_nonzero_15/Sumcond_12/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_15/Sum*
T0	
k
cond_12/truediv/CastCastcond_12/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
�
cond_12/truediv/Cast_1/SwitchSwitchcount_nonzero_17/Sumcond_12/pred_id*
T0	*
_output_shapes
: : *'
_class
loc:@count_nonzero_17/Sum
o
cond_12/truediv/Cast_1Castcond_12/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
i
cond_12/truedivRealDivcond_12/truediv/Castcond_12/truediv/Cast_1*
T0*
_output_shapes
: 
i
cond_12/ConstConst^cond_12/switch_f*
dtype0*
_output_shapes
: *
valueB 2        
b
cond_12/MergeMergecond_12/Constcond_12/truediv*
_output_shapes
: : *
T0*
N
V
recall_5/tagsConst*
valueB Brecall_5*
_output_shapes
: *
dtype0
X
recall_5ScalarSummaryrecall_5/tagscond_12/Merge*
_output_shapes
: *
T0
�
Const_7Const*�
value�B�	�"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                *
_output_shapes	
:�*
dtype0	
U
ArgMax_12/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
h
	ArgMax_12ArgMaxcond/Merge_1ArgMax_12/dimension*
_output_shapes	
:�*
T0*

Tidx0
K
Equal_18Equal	ArgMax_12Const_7*
T0	*
_output_shapes	
:�
Q
LogicalAnd_6
LogicalAndaccuracy/EqualEqual_18*
_output_shapes	
:�
]
count_nonzero_18/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
v
count_nonzero_18/NotEqualNotEqualLogicalAnd_6count_nonzero_18/NotEqual/y*
_output_shapes	
:�*
T0

p
count_nonzero_18/ToInt64Castcount_nonzero_18/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
`
count_nonzero_18/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
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
:�
]
count_nonzero_19/NotEqual/yConst*
value	B
 Z *
_output_shapes
: *
dtype0

r
count_nonzero_19/NotEqualNotEqualEqual_19count_nonzero_19/NotEqual/y*
_output_shapes	
:�*
T0

p
count_nonzero_19/ToInt64Castcount_nonzero_19/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
`
count_nonzero_19/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
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
:�*
T0*

Tidx0
K
Equal_20Equal	ArgMax_13Const_7*
T0	*
_output_shapes	
:�
]
count_nonzero_20/NotEqual/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
r
count_nonzero_20/NotEqualNotEqualEqual_20count_nonzero_20/NotEqual/y*
T0
*
_output_shapes	
:�
p
count_nonzero_20/ToInt64Castcount_nonzero_20/NotEqual*

SrcT0
*
_output_shapes	
:�*

DstT0	
`
count_nonzero_20/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
count_nonzero_20/SumSumcount_nonzero_20/ToInt64count_nonzero_20/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
N
Greater_12/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
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
cond_13/switch_tIdentitycond_13/Switch:1*
_output_shapes
: *
T0

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

�
cond_13/truediv/Cast/SwitchSwitchcount_nonzero_18/Sumcond_13/pred_id*'
_class
loc:@count_nonzero_18/Sum*
_output_shapes
: : *
T0	
k
cond_13/truediv/CastCastcond_13/truediv/Cast/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
�
cond_13/truediv/Cast_1/SwitchSwitchcount_nonzero_19/Sumcond_13/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_19/Sum*
T0	
o
cond_13/truediv/Cast_1Castcond_13/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
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
cond_13/MergeMergecond_13/Constcond_13/truediv*
_output_shapes
: : *
N*
T0
\
precision_6/tagsConst*
_output_shapes
: *
dtype0*
valueB Bprecision_6
^
precision_6ScalarSummaryprecision_6/tagscond_13/Merge*
T0*
_output_shapes
: 
N
Greater_13/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Z

Greater_13Greatercount_nonzero_20/SumGreater_13/y*
T0	*
_output_shapes
: 
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
�
cond_14/truediv/Cast/SwitchSwitchcount_nonzero_18/Sumcond_14/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_18/Sum*
T0	
k
cond_14/truediv/CastCastcond_14/truediv/Cast/Switch:1*

SrcT0	*
_output_shapes
: *

DstT0
�
cond_14/truediv/Cast_1/SwitchSwitchcount_nonzero_20/Sumcond_14/pred_id*
_output_shapes
: : *'
_class
loc:@count_nonzero_20/Sum*
T0	
o
cond_14/truediv/Cast_1Castcond_14/truediv/Cast_1/Switch:1*
_output_shapes
: *

DstT0*

SrcT0	
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
N*
T0
V
recall_6/tagsConst*
_output_shapes
: *
dtype0*
valueB Brecall_6
X
recall_6ScalarSummaryrecall_6/tagscond_14/Merge*
T0*
_output_shapes
: 
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
accuracy_1/tagsConst*
dtype0*
_output_shapes
: *
valueB B
accuracy_1
`

accuracy_1ScalarSummaryaccuracy_1/tagsaccuracy/accuracy*
_output_shapes
: *
T0
�
Merge/MergeSummaryMergeSummaryprecision_0recall_0precision_1recall_1precision_2recall_2precision_3recall_3precision_4recall_4precision_5recall_5precision_6recall_6
total_losslr
accuracy_1*
N*
_output_shapes
: 
^
total_loss_1/tagsConst*
valueB Btotal_loss_1*
_output_shapes
: *
dtype0
[
total_loss_1ScalarSummarytotal_loss_1/tagsloss/Sum*
T0*
_output_shapes
: 
N
	lr_1/tagsConst*
dtype0*
_output_shapes
: *
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
�

save/SaveV2/tensor_namesConst*�

value�	B�	$BVariableBbeta1_powerBbeta2_powerBencoder/initial_state_0Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1Binput_embedBinput_embed/AdamBinput_embed/Adam_1Boutput_projection/WBoutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/bBoutput_projection/b/AdamBoutput_projection/b/Adam_1*
dtype0*
_output_shapes
:$
�
save/SaveV2/shape_and_slicesConst*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariablebeta1_powerbeta2_powerencoder/initial_state_0encoder/initial_state_0/Adamencoder/initial_state_0/Adam_1encoder/initial_state_1encoder/initial_state_1/Adamencoder/initial_state_1/Adam_1encoder/initial_state_2encoder/initial_state_2/Adamencoder/initial_state_2/Adam_1encoder/initial_state_3encoder/initial_state_3/Adamencoder/initial_state_3/Adam_12encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_13encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_12encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_13encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1input_embedinput_embed/Adaminput_embed/Adam_1output_projection/Woutput_projection/W/Adamoutput_projection/W/Adam_1output_projection/boutput_projection/b/Adamoutput_projection/b/Adam_1*2
dtypes(
&2$
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
_class
loc:@save/Const*
T0
l
save/RestoreV2/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBBVariable
h
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/AssignAssignVariablesave/RestoreV2*
_class
loc:@Variable*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
q
save/RestoreV2_1/tensor_namesConst*
_output_shapes
:*
dtype0* 
valueBBbeta1_power
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_1Assignbeta1_powersave/RestoreV2_1*
_output_shapes
: *
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
q
save/RestoreV2_2/tensor_namesConst* 
valueBBbeta2_power*
_output_shapes
:*
dtype0
j
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_2Assignbeta2_powersave/RestoreV2_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@input_embed
}
save/RestoreV2_3/tensor_namesConst*,
value#B!Bencoder/initial_state_0*
_output_shapes
:*
dtype0
j
!save/RestoreV2_3/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assignencoder/initial_state_0save/RestoreV2_3*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_0*
validate_shape(*
_output_shapes
:	�
�
save/RestoreV2_4/tensor_namesConst*1
value(B&Bencoder/initial_state_0/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_4Assignencoder/initial_state_0/Adamsave/RestoreV2_4**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_5/tensor_namesConst*
_output_shapes
:*
dtype0*3
value*B(Bencoder/initial_state_0/Adam_1
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_5Assignencoder/initial_state_0/Adam_1save/RestoreV2_5*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_0
}
save/RestoreV2_6/tensor_namesConst*
_output_shapes
:*
dtype0*,
value#B!Bencoder/initial_state_1
j
!save/RestoreV2_6/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_6Assignencoder/initial_state_1save/RestoreV2_6*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	�
�
save/RestoreV2_7/tensor_namesConst*1
value(B&Bencoder/initial_state_1/Adam*
_output_shapes
:*
dtype0
j
!save/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_7Assignencoder/initial_state_1/Adamsave/RestoreV2_7*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_1
�
save/RestoreV2_8/tensor_namesConst*
_output_shapes
:*
dtype0*3
value*B(Bencoder/initial_state_1/Adam_1
j
!save/RestoreV2_8/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_8Assignencoder/initial_state_1/Adam_1save/RestoreV2_8**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
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
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9Assignencoder/initial_state_2save/RestoreV2_9**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
�
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
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_10Assignencoder/initial_state_2/Adamsave/RestoreV2_10*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�**
_class 
loc:@encoder/initial_state_2
�
save/RestoreV2_11/tensor_namesConst*3
value*B(Bencoder/initial_state_2/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_11Assignencoder/initial_state_2/Adam_1save/RestoreV2_11*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_2*
validate_shape(*
_output_shapes
:	�
~
save/RestoreV2_12/tensor_namesConst*,
value#B!Bencoder/initial_state_3*
dtype0*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_12Assignencoder/initial_state_3save/RestoreV2_12*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	�
�
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
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_13Assignencoder/initial_state_3/Adamsave/RestoreV2_13*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	�
�
save/RestoreV2_14/tensor_namesConst*3
value*B(Bencoder/initial_state_3/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_14/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assignencoder/initial_state_3/Adam_1save/RestoreV2_14*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	�
�
save/RestoreV2_15/tensor_namesConst*
_output_shapes
:*
dtype0*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_15Assign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasessave/RestoreV2_15*
_output_shapes	
:�*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
T0*
use_locking(
�
save/RestoreV2_16/tensor_namesConst*
dtype0*
_output_shapes
:*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam
k
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_16Assign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adamsave/RestoreV2_16*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
�
save/RestoreV2_17/tensor_namesConst*N
valueEBCB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_17Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1save/RestoreV2_17*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
validate_shape(*
_output_shapes	
:�
�
save/RestoreV2_18/tensor_namesConst*
_output_shapes
:*
dtype0*H
value?B=B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
k
"save/RestoreV2_18/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_18Assign3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightssave/RestoreV2_18* 
_output_shapes
:
��*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0*
use_locking(
�
save/RestoreV2_19/tensor_namesConst*
_output_shapes
:*
dtype0*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_19Assign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamsave/RestoreV2_19*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
��*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_20/tensor_namesConst*
dtype0*
_output_shapes
:*O
valueFBDB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1
k
"save/RestoreV2_20/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_20Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1save/RestoreV2_20*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
validate_shape(* 
_output_shapes
:
��
�
save/RestoreV2_21/tensor_namesConst*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_21Assign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasessave/RestoreV2_21*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
validate_shape(*
_output_shapes	
:�
�
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
�
save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_22Assign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adamsave/RestoreV2_22*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
validate_shape(*
_output_shapes	
:�
�
save/RestoreV2_23/tensor_namesConst*N
valueEBCB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_23Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1save/RestoreV2_23*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:�*
T0*
validate_shape(*
use_locking(
�
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
�
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_24Assign3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightssave/RestoreV2_24*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
validate_shape(* 
_output_shapes
:
��
�
save/RestoreV2_25/tensor_namesConst*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_25/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_25Assign8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adamsave/RestoreV2_25*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
��*
T0*
validate_shape(*
use_locking(
�
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
�
save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_26Assign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1save/RestoreV2_26*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
validate_shape(* 
_output_shapes
:
��
r
save/RestoreV2_27/tensor_namesConst*
_output_shapes
:*
dtype0* 
valueBBinput_embed
k
"save/RestoreV2_27/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_27Assigninput_embedsave/RestoreV2_27*
_class
loc:@input_embed*#
_output_shapes
:�*
T0*
validate_shape(*
use_locking(
w
save/RestoreV2_28/tensor_namesConst*
_output_shapes
:*
dtype0*%
valueBBinput_embed/Adam
k
"save/RestoreV2_28/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_28Assigninput_embed/Adamsave/RestoreV2_28*
_class
loc:@input_embed*#
_output_shapes
:�*
T0*
validate_shape(*
use_locking(
y
save/RestoreV2_29/tensor_namesConst*'
valueBBinput_embed/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_29Assigninput_embed/Adam_1save/RestoreV2_29*
use_locking(*
T0*
_class
loc:@input_embed*
validate_shape(*#
_output_shapes
:�
z
save/RestoreV2_30/tensor_namesConst*
dtype0*
_output_shapes
:*(
valueBBoutput_projection/W
k
"save/RestoreV2_30/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_30Assignoutput_projection/Wsave/RestoreV2_30*
_output_shapes
:	�*
validate_shape(*&
_class
loc:@output_projection/W*
T0*
use_locking(

save/RestoreV2_31/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Boutput_projection/W/Adam
k
"save/RestoreV2_31/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_31Assignoutput_projection/W/Adamsave/RestoreV2_31*
_output_shapes
:	�*
validate_shape(*&
_class
loc:@output_projection/W*
T0*
use_locking(
�
save/RestoreV2_32/tensor_namesConst*
_output_shapes
:*
dtype0*/
value&B$Boutput_projection/W/Adam_1
k
"save/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_32Assignoutput_projection/W/Adam_1save/RestoreV2_32*&
_class
loc:@output_projection/W*
_output_shapes
:	�*
T0*
validate_shape(*
use_locking(
z
save/RestoreV2_33/tensor_namesConst*(
valueBBoutput_projection/b*
dtype0*
_output_shapes
:
k
"save/RestoreV2_33/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_33Assignoutput_projection/bsave/RestoreV2_33*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(

save/RestoreV2_34/tensor_namesConst*-
value$B"Boutput_projection/b/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_34/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_34Assignoutput_projection/b/Adamsave/RestoreV2_34*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*&
_class
loc:@output_projection/b
�
save/RestoreV2_35/tensor_namesConst*
dtype0*
_output_shapes
:*/
value&B$Boutput_projection/b/Adam_1
k
"save/RestoreV2_35/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_35Assignoutput_projection/b/Adam_1save/RestoreV2_35*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedVariable*
_output_shapes
: *
dtype0*
_class
loc:@Variable
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedinput_embed*
_class
loc:@input_embed*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedencoder/initial_state_0*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_0
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedencoder/initial_state_1**
_class 
loc:@encoder/initial_state_1*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedencoder/initial_state_2**
_class 
loc:@encoder/initial_state_2*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedencoder/initial_state_3**
_class 
loc:@encoder/initial_state_3*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitialized2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedoutput_projection/W*
_output_shapes
: *
dtype0*&
_class
loc:@output_projection/W
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedoutput_projection/b*
_output_shapes
: *
dtype0*&
_class
loc:@output_projection/b
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedbeta1_power*
dtype0*
_output_shapes
: *
_class
loc:@input_embed
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedbeta2_power*
dtype0*
_output_shapes
: *
_class
loc:@input_embed
�
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedinput_embed/Adam*
dtype0*
_output_shapes
: *
_class
loc:@input_embed
�
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedinput_embed/Adam_1*
_class
loc:@input_embed*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedencoder/initial_state_0/Adam*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_0
�
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitializedencoder/initial_state_0/Adam_1*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_0
�
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitializedencoder/initial_state_1/Adam*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_1
�
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitializedencoder/initial_state_1/Adam_1*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_1
�
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitializedencoder/initial_state_2/Adam*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_2
�
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitializedencoder/initial_state_2/Adam_1**
_class 
loc:@encoder/initial_state_2*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitializedencoder/initial_state_3/Adam*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_3
�
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitializedencoder/initial_state_3/Adam_1*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_3
�
7report_uninitialized_variables/IsVariableInitialized_24IsVariableInitialized8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_25IsVariableInitialized:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1*
_output_shapes
: *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
�
7report_uninitialized_variables/IsVariableInitialized_26IsVariableInitialized7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_27IsVariableInitialized9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
�
7report_uninitialized_variables/IsVariableInitialized_28IsVariableInitialized8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
�
7report_uninitialized_variables/IsVariableInitialized_29IsVariableInitialized:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_30IsVariableInitialized7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_31IsVariableInitialized9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
�
7report_uninitialized_variables/IsVariableInitialized_32IsVariableInitializedoutput_projection/W/Adam*
dtype0*
_output_shapes
: *&
_class
loc:@output_projection/W
�
7report_uninitialized_variables/IsVariableInitialized_33IsVariableInitializedoutput_projection/W/Adam_1*
_output_shapes
: *
dtype0*&
_class
loc:@output_projection/W
�
7report_uninitialized_variables/IsVariableInitialized_34IsVariableInitializedoutput_projection/b/Adam*
dtype0*
_output_shapes
: *&
_class
loc:@output_projection/b
�
7report_uninitialized_variables/IsVariableInitialized_35IsVariableInitializedoutput_projection/b/Adam_1*
dtype0*
_output_shapes
: *&
_class
loc:@output_projection/b
�
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_217report_uninitialized_variables/IsVariableInitialized_227report_uninitialized_variables/IsVariableInitialized_237report_uninitialized_variables/IsVariableInitialized_247report_uninitialized_variables/IsVariableInitialized_257report_uninitialized_variables/IsVariableInitialized_267report_uninitialized_variables/IsVariableInitialized_277report_uninitialized_variables/IsVariableInitialized_287report_uninitialized_variables/IsVariableInitialized_297report_uninitialized_variables/IsVariableInitialized_307report_uninitialized_variables/IsVariableInitialized_317report_uninitialized_variables/IsVariableInitialized_327report_uninitialized_variables/IsVariableInitialized_337report_uninitialized_variables/IsVariableInitialized_347report_uninitialized_variables/IsVariableInitialized_35*
_output_shapes
:$*
N$*

axis *
T0

y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:$
�

$report_uninitialized_variables/ConstConst*
dtype0*
_output_shapes
:$*�

value�	B�	$BVariableBinput_embedBencoder/initial_state_0Bencoder/initial_state_1Bencoder/initial_state_2Bencoder/initial_state_3B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesBoutput_projection/WBoutput_projection/bBbeta1_powerBbeta2_powerBinput_embed/AdamBinput_embed/Adam_1Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1B7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1Boutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/b/AdamBoutput_projection/b/Adam_1
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:$*
dtype0*
_output_shapes
:
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
:
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:$
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
valueB:*
_output_shapes
:*
dtype0
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
_output_shapes
: *
end_mask*
new_axis_mask *
ellipsis_mask *

begin_mask *
shrink_axis_mask *
T0*
Index0
�
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*

axis *
_output_shapes
:*
T0*
N
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
T0*
Tshape0*
_output_shapes
:$
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
T0
*
_output_shapes
:$*
Tshape0
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*#
_output_shapes
:���������*
T0	*
squeeze_dims

�
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*#
_output_shapes
:���������*
validate_indices(*
Tparams0*
Tindices0	
�
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
init"�O
cond_context�N�N
�
batch_and_pad/cond/cond_textbatch_and_pad/cond/pred_id:0batch_and_pad/cond/switch_t:0 *�
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
random_queue_test_Dequeue:2W
random_queue_test_Dequeue:28batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_3:1\
"batch_and_pad/padding_fifo_queue:06batch_and_pad/cond/padding_fifo_queue_enqueue/Switch:1W
random_queue_test_Dequeue:18batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_2:1W
random_queue_test_Dequeue:08batch_and_pad/cond/padding_fifo_queue_enqueue/Switch_1:1
�
batch_and_pad/cond/cond_text_1batch_and_pad/cond/pred_id:0batch_and_pad/cond/switch_f:0*h
)batch_and_pad/cond/control_dependency_1:0
batch_and_pad/cond/pred_id:0
batch_and_pad/cond/switch_f:0
�
batch_and_pad_1/cond/cond_textbatch_and_pad_1/cond/pred_id:0batch_and_pad_1/cond/switch_t:0 *�
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
�
 batch_and_pad_1/cond/cond_text_1batch_and_pad_1/cond/pred_id:0batch_and_pad_1/cond/switch_f:0*n
+batch_and_pad_1/cond/control_dependency_1:0
batch_and_pad_1/cond/pred_id:0
batch_and_pad_1/cond/switch_f:0
�
cond/cond_textcond/pred_id:0cond/switch_t:0 *�
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
�
cond/cond_text_1cond/pred_id:0cond/switch_f:0*�
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
�
cond_1/cond_textcond_1/pred_id:0cond_1/switch_t:0 *�
cond_1/pred_id:0
cond_1/switch_t:0
cond_1/truediv/Cast/Switch:1
cond_1/truediv/Cast:0
cond_1/truediv/Cast_1/Switch:1
cond_1/truediv/Cast_1:0
cond_1/truediv:0
count_nonzero/Sum:0
count_nonzero_1/Sum:07
count_nonzero_1/Sum:0cond_1/truediv/Cast_1/Switch:13
count_nonzero/Sum:0cond_1/truediv/Cast/Switch:1
p
cond_1/cond_text_1cond_1/pred_id:0cond_1/switch_f:0*5
cond_1/Const:0
cond_1/pred_id:0
cond_1/switch_f:0
�
cond_2/cond_textcond_2/pred_id:0cond_2/switch_t:0 *�
cond_2/pred_id:0
cond_2/switch_t:0
cond_2/truediv/Cast/Switch:1
cond_2/truediv/Cast:0
cond_2/truediv/Cast_1/Switch:1
cond_2/truediv/Cast_1:0
cond_2/truediv:0
count_nonzero/Sum:0
count_nonzero_2/Sum:07
count_nonzero_2/Sum:0cond_2/truediv/Cast_1/Switch:13
count_nonzero/Sum:0cond_2/truediv/Cast/Switch:1
p
cond_2/cond_text_1cond_2/pred_id:0cond_2/switch_f:0*5
cond_2/Const:0
cond_2/pred_id:0
cond_2/switch_f:0
�
cond_3/cond_textcond_3/pred_id:0cond_3/switch_t:0 *�
cond_3/pred_id:0
cond_3/switch_t:0
cond_3/truediv/Cast/Switch:1
cond_3/truediv/Cast:0
cond_3/truediv/Cast_1/Switch:1
cond_3/truediv/Cast_1:0
cond_3/truediv:0
count_nonzero_3/Sum:0
count_nonzero_4/Sum:07
count_nonzero_4/Sum:0cond_3/truediv/Cast_1/Switch:15
count_nonzero_3/Sum:0cond_3/truediv/Cast/Switch:1
p
cond_3/cond_text_1cond_3/pred_id:0cond_3/switch_f:0*5
cond_3/Const:0
cond_3/pred_id:0
cond_3/switch_f:0
�
cond_4/cond_textcond_4/pred_id:0cond_4/switch_t:0 *�
cond_4/pred_id:0
cond_4/switch_t:0
cond_4/truediv/Cast/Switch:1
cond_4/truediv/Cast:0
cond_4/truediv/Cast_1/Switch:1
cond_4/truediv/Cast_1:0
cond_4/truediv:0
count_nonzero_3/Sum:0
count_nonzero_5/Sum:07
count_nonzero_5/Sum:0cond_4/truediv/Cast_1/Switch:15
count_nonzero_3/Sum:0cond_4/truediv/Cast/Switch:1
p
cond_4/cond_text_1cond_4/pred_id:0cond_4/switch_f:0*5
cond_4/Const:0
cond_4/pred_id:0
cond_4/switch_f:0
�
cond_5/cond_textcond_5/pred_id:0cond_5/switch_t:0 *�
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
�
cond_6/cond_textcond_6/pred_id:0cond_6/switch_t:0 *�
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
�
cond_7/cond_textcond_7/pred_id:0cond_7/switch_t:0 *�
cond_7/pred_id:0
cond_7/switch_t:0
cond_7/truediv/Cast/Switch:1
cond_7/truediv/Cast:0
cond_7/truediv/Cast_1/Switch:1
cond_7/truediv/Cast_1:0
cond_7/truediv:0
count_nonzero_10/Sum:0
count_nonzero_9/Sum:05
count_nonzero_9/Sum:0cond_7/truediv/Cast/Switch:18
count_nonzero_10/Sum:0cond_7/truediv/Cast_1/Switch:1
p
cond_7/cond_text_1cond_7/pred_id:0cond_7/switch_f:0*5
cond_7/Const:0
cond_7/pred_id:0
cond_7/switch_f:0
�
cond_8/cond_textcond_8/pred_id:0cond_8/switch_t:0 *�
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
�
cond_9/cond_textcond_9/pred_id:0cond_9/switch_t:0 *�
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
�
cond_10/cond_textcond_10/pred_id:0cond_10/switch_t:0 *�
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
�
cond_11/cond_textcond_11/pred_id:0cond_11/switch_t:0 *�
cond_11/pred_id:0
cond_11/switch_t:0
cond_11/truediv/Cast/Switch:1
cond_11/truediv/Cast:0
cond_11/truediv/Cast_1/Switch:1
cond_11/truediv/Cast_1:0
cond_11/truediv:0
count_nonzero_15/Sum:0
count_nonzero_16/Sum:07
count_nonzero_15/Sum:0cond_11/truediv/Cast/Switch:19
count_nonzero_16/Sum:0cond_11/truediv/Cast_1/Switch:1
v
cond_11/cond_text_1cond_11/pred_id:0cond_11/switch_f:0*8
cond_11/Const:0
cond_11/pred_id:0
cond_11/switch_f:0
�
cond_12/cond_textcond_12/pred_id:0cond_12/switch_t:0 *�
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
�
cond_13/cond_textcond_13/pred_id:0cond_13/switch_t:0 *�
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
�
cond_14/cond_textcond_14/pred_id:0cond_14/switch_t:0 *�
cond_14/pred_id:0
cond_14/switch_t:0
cond_14/truediv/Cast/Switch:1
cond_14/truediv/Cast:0
cond_14/truediv/Cast_1/Switch:1
cond_14/truediv/Cast_1:0
cond_14/truediv:0
count_nonzero_18/Sum:0
count_nonzero_20/Sum:07
count_nonzero_18/Sum:0cond_14/truediv/Cast/Switch:19
count_nonzero_20/Sum:0cond_14/truediv/Cast_1/Switch:1
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
4report_uninitialized_variables/boolean_mask/Gather:0"�
	summaries�
�
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
lr_1:0"�

trainable_variables�
�

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
�
5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights:0:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read:0
�
4encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases:09encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read:0
�
5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights:0:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Assign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read:0
�
4encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases:09encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/read:0
O
output_projection/W:0output_projection/W/Assignoutput_projection/W/read:0
O
output_projection/b:0output_projection/b/Assignoutput_projection/b/read:0"�
queue_runners��
�
 batch_and_pad/padding_fifo_queuebatch_and_pad/cond/Merge:0&batch_and_pad/padding_fifo_queue_Close"(batch_and_pad/padding_fifo_queue_Close_1*
�
"batch_and_pad_1/padding_fifo_queuebatch_and_pad_1/cond/Merge:0(batch_and_pad_1/padding_fifo_queue_Close"*batch_and_pad_1/padding_fifo_queue_Close_1*"
local_init_op


group_deps"��
while_context����
��
(encoder_1/rnn/while/encoder_1/rnn/while/ *encoder_1/rnn/while/LoopCond:02encoder_1/rnn/while/Merge:0:encoder_1/rnn/while/Identity:0Bencoder_1/rnn/while/Exit:0Bencoder_1/rnn/while/Exit_1:0Bencoder_1/rnn/while/Exit_2:0Bencoder_1/rnn/while/Exit_3:0Bencoder_1/rnn/while/Exit_4:0Bencoder_1/rnn/while/Exit_5:0Bgradients/f_count_2:0J��
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
gradients/f_count_2:0�
>gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc:0Agradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnter:0�
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnter:0�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc:0Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnter:0;
encoder_1/rnn/zeros:0"encoder_1/rnn/while/Select/Enter:0{
:gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc:0=gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnter:0{
:gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc:0=gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnter:0�
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc:0bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnter:0}
Jencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0/encoder_1/rnn/while/TensorArrayReadV3/Enter_1:0^
encoder_1/rnn/TensorArray:0?encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0�
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read:0Lencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter:0C
encoder_1/rnn/strided_slice_2:0 encoder_1/rnn/while/Less/Enter:0G
encoder_1/rnn/CheckSeqLen:0(encoder_1/rnn/while/GreaterEqual/Enter:0�
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc:0bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnter:0�
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc:0egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter:0�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc:0Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnter:0{
:gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc:0=gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnter:0N
encoder_1/rnn/TensorArray_1:0-encoder_1/rnn/while/TensorArrayReadV3/Enter:0�
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/read:0Cencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter:0�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnter:0�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnter:0�
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read:0Cencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter:0�
`gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc:0cgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnter:0�
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc:0Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnter:0�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc:0Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnter:0�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc:0Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnter:0�
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read:0Lencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter:0�
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc:0ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnter:0�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnter:0�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnter:0�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnter:0�
>gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc:0Agradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnter:0{
:gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc:0=gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnter:0�
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc:0ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnter:0�
>gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc:0Agradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnter:0�
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnter:0�
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnter:0�
>gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc:0Agradients/encoder_1/rnn/while/Select_2_grad/zeros_like/RefEnter:0�
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc:0egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter:0�
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc:0Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnter:0"�"
	variables�"�"
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
�
5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights:0:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read:0
�
4encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases:09encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read:0
�
5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights:0:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Assign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read:0
�
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
�
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam:0?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/Assign?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/read:0
�
<encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1:0Aencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/AssignAencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/read:0
�
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam:0>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/Assign>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/read:0
�
;encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1:0@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/Assign@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/read:0
�
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam:0?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/Assign?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/read:0
�
<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1:0Aencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/AssignAencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/read:0
�
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam:0>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/Assign>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/read:0
�
;encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1:0@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/Assign@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/read:0
^
output_projection/W/Adam:0output_projection/W/Adam/Assignoutput_projection/W/Adam/read:0
d
output_projection/W/Adam_1:0!output_projection/W/Adam_1/Assign!output_projection/W/Adam_1/read:0
^
output_projection/b/Adam:0output_projection/b/Adam/Assignoutput_projection/b/Adam/read:0
d
output_projection/b/Adam_1:0!output_projection/b/Adam_1/Assign!output_projection/b/Adam_1/read:0_ֹr       �m�.	S"L�C�A��:v2ĥ$       B+�M	��"L�C�A��*

Variable/sec    z�ڐ:       ���	��1L�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�]�$       B+�M	fK"��C�A��*

Variable/sec|g�?I�Ŋ:       ���	�@��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt<�w;$       B+�M	<�"��C�A��*

Variable/secl�?G�n�:       ���	I�l��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpti�'C$       B+�M	�Y"-�C�A��*

Variable/seco��?��:       ���	��T-�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt!S+�$       B+�M	�H"x�C�A��*

Variable/sec'׃?P�,�:       ���	��Ux�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�@&u$       B+�M	�L"��C�A��*

Variable/sec׃?��:       ���	��K��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt7's$       B+�M	�E"�C�A��*

Variable/sec���?�xh�:       ���	�=�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt=�O�$       B+�M	�P"Y�C�A��*

Variable/sec�i�?�9R2:       ���	�\Y�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt}�$       B+�M	�o"��C�A��*

Variable/sec+��?/���:       ���	��F��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt1�$       B+�M	�f"��C�A��*

Variable/sec�?��G�:       ���	�@��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt,��$       B+�M	�]":�C�A��*

Variable/sec �?k�Tr:       ���	�U:�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptG�f$       B+�M	&T"��C�A��*

Variable/secܻ{?�y�a:       ���	��H��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��3$       B+�M	�|"��C�A��*

Variable/sec�l�?�O�:       ���	 y;��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�^�$       B+�M	)]"�C�A��*

Variable/sec��?S���:       ���	!Y�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt=5��$       B+�M	�C"f�C�A��*

Variable/sec��?te�:       ���	9�<f�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt%!��$       B+�M	�i~��C�A��*

Variable/sec���>_��B:       ���	"4���C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt\�3,$       B+�M	/P"��C�A��*

Variable/sec	M{?�cq�:       ���	!�>��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��֢$       B+�M	Z"G�C�A��*

Variable/secN��?��&�:       ���	H�BG�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	�J"��C�A��*

Variable/sec;"�?���:       ���	��K��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt����$       B+�M	^"��C�A��*

Variable/sec�Qx?�&�?:       ���	�B��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptP=�$       B+�M	�P"(�C�A��*

Variable/sec�ww?�P�G:       ���	��;(�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt����$       B+�M	8L"s�C�A��*

Variable/sec�Xr?~I:       ���	o>s�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M	Ve"��C�A��*

Variable/sec��p?�P�:       ���	��V��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt0b�$       B+�M	Nc"	�C�A��*

Variable/sec�_l?�L:       ���	��?	�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��3$       B+�M	�V"T�C�A��*

Variable/secy�z?��0�:       ���	��NT�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��K$       B+�M	�d"��C�A��*

Variable/sec˴�?�z�D:       ���	[a`��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�u$       B+�M	^H"��C�A��*

Variable/sec1 �?��$:       ���	�Q��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptq��C$       B+�M	�T"5�C�A��*

Variable/secӴ�?�7��:       ���	f�?5�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�8"$       B+�M	P"��C�A�*

Variable/sec~ڀ?�*-�:       ���	�#D��C�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckptu�$       B+�M	SY"��C�A��*

Variable/secٴ�?3�~:       ���	��;��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptCN%�$       B+�M	_"�C�Aˌ*

Variable/sec"�?M�:       ���	CJ�C�A̌:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��*�$       B+�M	��#a�C�A��*

Variable/secf�?�e��:       ���	��<a�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�N�$       B+�M	�u"��C�A��*

Variable/secz��?����:       ���	�`>��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�W�}$       B+�M	cC"��C�Aۓ*

Variable/secH�?�iMp:       ���	�@��C�Aܓ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	�#B�C�A��*

Variable/sec��z?��{�:       ���	(AB�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�3�$       B+�M	ׇ"��C�A��*

Variable/sec�Sx?�w,�:       ���	I-@��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt.H�$       B+�M	�k"��C�AϚ*

Variable/sec�%?�e(�:       ���	�<��C�AϚ:+'logs/s2l_2017-05-06_13-29-45/model.ckptS�x$       B+�M	�]"#�C�A��*

Variable/sec���?Y�{$:       ���	"E#�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptt ��$       B+�M	 S"n�C�A��*

Variable/sec�ڀ?R'1:       ���	�;n�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�ç[$       B+�M	�q"��C�A�*

Variable/sece��?)��:       ���	,@��C�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��9$       B+�M	�W"�C�A��*

Variable/secM"�?;#Č:       ���	�ZS�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��c$       B+�M	��#O�C�AƦ*

Variable/sec挂?I��T:       ���	KO�C�AƦ:+'logs/s2l_2017-05-06_13-29-45/model.ckptA��R$       B+�M	�e"��C�A��*

Variable/sec���?ք��:       ���	2�=��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��A�$       B+�M	�W"��C�A��*

Variable/sec$Rx?��١:       ���	}�S��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpty鏙$       B+�M	�c"0�C�A��*

Variable/sec�Qx?h/�:       ���	�;0�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M	q"{�C�Aد*

Variable/sec1�o?U�	:       ���	�<{�C�Aد:+'logs/s2l_2017-05-06_13-29-45/model.ckpt ��r$       B+�M	Me"��C�A��*

Variable/secV�|?�g��:       ���	�'M��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptF�_$       B+�M	#L"�C�A��*

Variable/sec�{?�0�y:       ���	MJB�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt	c̜$       B+�M	FC"\�C�AͶ*

Variable/secc�z?��@|:       ���	�I\�C�Aζ:+'logs/s2l_2017-05-06_13-29-45/model.ckptҥ�$       B+�M	�w"��C�A��*

Variable/sec�l�?��*�:       ���	��<��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt'�v$       B+�M	�n"��C�A��*

Variable/sec."�?��B
:       ���	!�F��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt͆'$       B+�M	E%=�C�Aݽ*

Variable/secp��?���::       ���	|CG=�C�A޽:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�$       B+�M	�M"��C�A��*

Variable/sec
��?8�D�:       ���	S�J��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	�x"��C�A��*

Variable/sec��?����:       ���	��B��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptײ��$       B+�M	��"�C�A��*

Variable/secд�?ÂQ�:       ���	��Q�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptu���$       B+�M	�I"i�C�A��*

Variable/secƏ�?�E�a:       ���	
=i�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�
#�$       B+�M	]Q"��C�A��*

Variable/secK��?�z�:       ���	��C��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt����$       B+�M	Yj"��C�A��*

Variable/sec1��?�_t�:       ���	�A��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	_\"J�C�A��*

Variable/secHK~?�2�:       ���	�<J�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��'�$       B+�M	�q"��C�A��*

Variable/sec �z?�j��:       ���	s�;��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�D�N$       B+�M	�S"��C�A��*

Variable/secq}?P�QS:       ���	P��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt_�R$       B+�M	)_"+�C�A��*

Variable/sec̴�?�Z��:       ���	�B;+�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�P$       B+�M	�T"v�C�A��*

Variable/secQm�?)�8:       ���	�Fv�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���~$       B+�M	*W"��C�A��*

Variable/secS��?aiB�:       ���	��<��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptv�Ĺ$       B+�M	��"�C�A��*

Variable/sec��?��B�:       ���	H�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt]uG�$       B+�M	5C"W�C�A��*

Variable/secΏ�?��у:       ���	}#SW�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt,$       B+�M	EK"��C�A��*

Variable/secN��?Id.�:       ���	��F��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptՠA$       B+�M	�w"��C�A��*

Variable/sec��z?�	FK:       ���	� C��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�L$       B+�M	eV"8�C�A��*

Variable/sec�~q?#��:       ���	�>8�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	$D"��C�A��*

Variable/secn�i?����:       ���	�uA��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt5��2$       B+�M	d"��C�A��*

Variable/sec?ڀ?.���:       ���	 �<��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�K$y$       B+�M	�^"�C�A��*

Variable/sec׃?���:       ���	0�K�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptnH\�$       B+�M	*R"d�C�A��*

Variable/sec5"�?C�:       ���	o�=d�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptp�8$       B+�M	�J"��C�A��*

Variable/secg��?�	��:       ���	-=��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�ʗ�$       B+�M	�_"��C�A��*

Variable/sec�G�?�:n�:       ���	�n=��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt3�$       B+�M	pD"E�C�A��*

Variable/sec���?�=5�:       ���	�?<E�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt-�/�$       B+�M	�L"��C�A��*

Variable/sec۴�?%l:       ���	K;E��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptܚ�$       B+�M	�f"��C�A��*

Variable/sec�!�?6�}d:       ���	(eC��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M	�E"&�C�A��*

Variable/sec���?���:       ���	�iH&�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�$M$$       B+�M	�d"q�C�A��*

Variable/sec���?��po:       ���	�:Mq�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpti�$       B+�M	ٯ"��C�A��*

Variable/secú{?��/:       ���	OI��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�Kʣ$       B+�M	H"�C�A��*

Variable/sec_H�?e��:       ���	��;�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptL�:�$       B+�M		N"R�C�A�*

Variable/sec"�?�/h:       ���	�ONR�C�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��Fu$       B+�M	1_"��C�A��*

Variable/secx��?�+U:       ���	�?��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M	aP"��C�A΋*

Variable/sec�i�?�%�:       ���	HNi��C�Aϋ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt����$       B+�M	"3�C�A��*

Variable/sec�!�?�0i�:       ���	�>3�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt!�0�$       B+�M	�e"~�C�A��*

Variable/secxm�?i�:�:       ���	��E~�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��
Y$       B+�M	l"��C�Aߒ*

Variable/sec���?���:       ���	�Q��C�Aߒ:+'logs/s2l_2017-05-06_13-29-45/model.ckptT�O$       B+�M	[`"�C�A��*

Variable/sec���?<���:       ���	{�F�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt>Օ$       B+�M	@i"_�C�A×*

Variable/sec"�?����:       ���	��Z_�C�Aė:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M	YR"��C�A��*

Variable/secJ"�?f�w:       ���	OB��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt,d�$       B+�M	�Z"��C�A��*

Variable/sec�v?�@Ϧ:       ���	��_��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�u��$       B+�M	ZJ"@�C�A��*

Variable/sec�z?!*U:       ���	��>@�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	�^"��C�A̠*

Variable/sec��h?�F/�:       ���	2�B��C�A͠:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�s��$       B+�M	GW"��C�A��*

Variable/seci��?��0q:       ���	��A��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��.�$       B+�M	�B"!�C�A��*

Variable/sec���?{�U&:       ���	��G!�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpto$       B+�M	�u"l�C�A�*

Variable/sec��?��:       ���	f�Cl�C�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��=$       B+�M	�]"��C�A��*

Variable/sec��?-R-�:       ���	�L��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	�$�C�A¬*

Variable/sec��?�rNR:       ���	XB�C�Aì:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M	�H"M�C�A��*

Variable/secd��?��^:       ���	�<M�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptw߇U$       B+�M	vQ"��C�A��*

Variable/sec���?h��:       ���	�r<��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	^L"��C�Aڳ*

Variable/sec���?t#|:       ���	"�G��C�A۳:+'logs/s2l_2017-05-06_13-29-45/model.ckpt߽U�$       B+�M	".�C�A��*

Variable/sec��?�[:       ���	_~N.�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptv�'�$       B+�M	`["y�C�A��*

Variable/secj�?�w
:       ���	aRAy�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptV��$       B+�M	�$��C�A�*

Variable/sec�|?��b�:       ���	�,?��C�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��G<$       B+�M	we"�C�A��*

Variable/secm3y?��:       ���	3�N�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	h�$Z�C�A��*

Variable/secӃ?9מ>:       ���	ǡ<Z�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt����$       B+�M	�|#��C�A��*

Variable/secy��?�x�P:       ���	�O��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt2$       B+�M	�N"��C�A��*

Variable/sec���?CP::       ���	�@U��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��.U$       B+�M	Dj";�C�A��*

Variable/sech��?o�:       ���	�N;�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptKg$       B+�M	��#��C�A��*

Variable/secB��?%-�m:       ���	G<E��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��h)$       B+�M	�r"��C�A��*

Variable/sec�k�?���d:       ���	�K��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�>2�$       B+�M	~R"�C�A��*

Variable/sec
j�?^
�:       ���	��T�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��#]$       B+�M	\X"g�C�A��*

Variable/sec�i�?	S�_:       ���	��Rg�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�)/$       B+�M	�k"��C�A��*

Variable/sect��?a�:       ���	΋L��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�)� $       B+�M	QK"��C�A��*

Variable/secj�v?
���:       ���	�C��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt8s��$       B+�M	�a"H�C�A��*

Variable/secwXr?LUߙ:       ���	QhFH�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��ނ$       B+�M	�N"��C�A��*

Variable/sec�ww?���1:       ���	QgD��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�+�$       B+�M	?S"��C�A��*

Variable/sec�i�?|{R�:       ���	� I��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt4�j�$       B+�M	x^")�C�A��*

Variable/sec�i�?t�:       ���	�0c)�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptgb�K$       B+�M	�E"t�C�A��*

Variable/secK"�?˛�X:       ���	sh>t�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��7�$       B+�M	^K"��C�A��*

Variable/sec׃?Z�*p:       ���	ՓG��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptYp$�$       B+�M	}X"
�C�A��*

Variable/sec�G�?�%�:       ���	�MO
�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�n�$       B+�M	�J"U�C�A��*

Variable/sec�i�?�x$S:       ���	@U�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�Tuq$       B+�M	�J"��C�A��*

Variable/sec���?XtL:       ���	�PO��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptK���$       B+�M	S"��C�A��*

Variable/sec�/�?U��:       ���	?��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M		�"6�C�A��*

Variable/sec"�?�QW?:       ���	G<76�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�Ob$       B+�M	�W"��C�A��*

Variable/sec�
�?/\4&:       ���	�18��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt4��$       B+�M	�r"��C�A��*

Variable/sec�!�?I$BG:       ���	:"@��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptu��$       B+�M	f"�C�A��*

Variable/secU��?Y�:       ���	��@�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�UX3$       B+�M	�"b�C�A��*

Variable/sec�?�J=:       ���	�!Sb�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�Ga$       B+�M	�V"��C�A��*

Variable/secG�?k! �:       ���	��6��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt:ڻ�$       B+�M	�Q"��C�A��*

Variable/secL
�?��@:       ���	�5��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt0t��$       B+�M	�b"C�C�A��*

Variable/sec�G�?�$��:       ���	s�<C�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt;��J$       B+�M	t_"��C�A��*

Variable/sec�i�?F<k|:       ���	��L��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���<$       B+�M	�U"��C�A��*

Variable/sec�z�?-�
�:       ���	�+P��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�[�$       B+�M	�D"$�C�Aڏ*

Variable/sec���?����:       ���	�,C$�C�Aڏ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�%�E$       B+�M	�`"o�C�A��*

Variable/sec�z�?-aP:       ���	�_8o�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���l$       B+�M	K["��C�A��*

Variable/sec˻�?0Ub�:       ���	!9��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�"+�$       B+�M	�o"�C�A��*

Variable/sec&�?���F:       ���	�?�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptP�E$       B+�M	�P"P�C�A��*

Variable/sec���?csJ�:       ���	�19P�C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt^�Y'$       B+�M	�g"��C�Aٞ*

Variable/secw�?T�$:       ���	�M5��C�Aڞ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt<��$       B+�M	�l"��C�A��*

Variable/sec�?���:       ���	��T��C�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��v3$       B+�M	�I$1 D�A��*

Variable/sec��?=�9�:       ���	y61 D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�x F$       B+�M	@�"| D�A�*

Variable/sec�{�?�fk:       ���	=�=| D�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt.���$       B+�M	4d"� D�A��*

Variable/sec�
�?��ZV:       ���	n8� D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptQ#�$       B+�M	OZ"D�A��*

Variable/sec�/�?�{j�:       ���	��@D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	��"]D�A�*

Variable/sec!v�?v��~:       ���	�;]D�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt:�A�$       B+�M	�W"�D�A��*

Variable/sec��?MKO�:       ���	DN?�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��A&$       B+�M	�R"�D�A��*

Variable/sec�w�?#�A:       ���	��:�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt9�$       B+�M	�p">D�A��*

Variable/sec[��?�O��:       ���	�U=>D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��S	$       B+�M	d�"�D�A��*

Variable/sec?H#u::       ���	�S>�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�m��$       B+�M	�W"�D�A��*

Variable/secV�?�p�k:       ���	��I�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptz$�l$       B+�M	p�"D�A��*

Variable/sec?�A9�:       ���	��TD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�/u$       B+�M	B{"jD�A��*

Variable/sec�U�?;8��:       ���	�z<jD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptBR^�$       B+�M	�t"�D�A��*

Variable/sec�z�?��|�:       ���	�k;�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�оJ$       B+�M	�c" D�A��*

Variable/secw�?4���:       ���	
�: D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�h2o$       B+�M	�O"KD�A��*

Variable/sec��?m�h#:       ���	�>KD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt?3$,$       B+�M	0F"�D�A��*

Variable/sec��?K&~~:       ���	]=�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptj���$       B+�M	zP"�D�A��*

Variable/secKU�?5�E:       ���	�q:�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�,�7$       B+�M	~t",D�A��*

Variable/secU�?�~[7:       ���	�L,D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	,D"wD�A��*

Variable/secy�?ڼ��:       ���	�/KwD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��2w$       B+�M	NE"�D�A��*

Variable/sec�?.�G:       ���	��9�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�o|=$       B+�M	@L"D�A��*

Variable/sec��?��:       ���	��>D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptd�J�$       B+�M	(d"XD�A��*

Variable/sec��?u�I:       ���	��:XD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��F|$       B+�M	V~"�D�A��*

Variable/sec��?k�~:       ���	��F�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt2(�$       B+�M	am"�D�A��*

Variable/sec�?�4':       ���	�fO�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�OH�$       B+�M	�)$9D�A��*

Variable/sec�X�?X�}�:       ���	�9E9D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�]��$       B+�M	@O"�D�A��*

Variable/sec��?D�z$:       ���	o�E�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt �;�$       B+�M	�b"�D�A��*

Variable/seci? ��:       ���	aA�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt&�r�$       B+�M	�g"D�A��*

Variable/sec�?{�7:       ���	�=D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��\$       B+�M	�l"eD�A��*

Variable/sec�/�?��B:       ���	��XeD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt{,��$       B+�M	2q"�D�A��*

Variable/sec�?�t��:       ���	3m;�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M	�j"�D�A��*

Variable/sec�?{��.:       ���	�EA�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt+���$       B+�M	�X"F	D�A��*

Variable/sec�?4��q:       ���	g8F	D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�1$       B+�M	�\"�	D�A��*

Variable/sec�?���W:       ���	��@�	D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�R��$       B+�M	6u"�	D�Ał*

Variable/sec?��?U�.�:       ���	��B�	D�AƂ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	�W"'
D�A��*

Variable/sec%�?d] :       ���	�iS'
D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��R$       B+�M	�I"r
D�A�*

Variable/secC\�?��97:       ���	��;r
D�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�q�$       B+�M	�\"�
D�A��*

Variable/sec��?x_�:       ���	:�A�
D�A:+'logs/s2l_2017-05-06_13-29-45/model.ckpt.� c$       B+�M	�Z"D�A��*

Variable/sec�?`y-�:       ���	q�GD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt[���$       B+�M	�Y"SD�A��*

Variable/secUU�?�V(:       ���	�>BSD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt����$       B+�M	6v"�D�Aג*

Variable/sec�2�?�s�F:       ���	�O�D�Aؒ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�
�$       B+�M	of"�D�A��*

Variable/secvU�?�wi:       ���	�B\�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��f$       B+�M	7U"4D�A��*

Variable/sec{�?|$�:       ���	�G;4D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt{*�F$       B+�M	�^"D�A�*

Variable/sec>U�?�R�:       ���	V�9D�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�cF�$       B+�M	_B"�D�A͝*

Variable/sec�U�?��<:       ���	((;�D�A͝:+'logs/s2l_2017-05-06_13-29-45/model.ckpt%F�$       B+�M	H4$D�A��*

Variable/sec�̉?�;:       ���	��>D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt"xdw$       B+�M	�H"`D�A�*

Variable/sec1��?aD:       ���	��9`D�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�F�$       B+�M	��#�D�A��*

Variable/sec��?�w�:       ���	��F�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt^�3�$       B+�M	jP"�D�A��*

Variable/sec���?2vv:       ���	�:�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�>�g$       B+�M	F#AD�A�*

Variable/secpS�?:�Ȗ:       ���	+�CAD�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���4$       B+�M	rj"�D�Aǭ*

Variable/sec��?�6a�:       ���	i�@�D�Aȭ:+'logs/s2l_2017-05-06_13-29-45/model.ckpti<�Y$       B+�M	c�#�D�A��*

Variable/sec���?DҊ�:       ���	�<�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptN�;�$       B+�M	�D""D�A��*

Variable/sec/~�?xmB�:       ���	>�A"D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�X��$       B+�M	�X"mD�A�*

Variable/sec1U�?=s:       ���	�h;mD�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt,e��$       B+�M	�L"�D�A��*

Variable/sec�?�m:       ���	�B�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��8$       B+�M	O\"D�A��*

Variable/sec9U�?KO�^:       ���	E�RD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt3,-$       B+�M	�["ND�A��*

Variable/sec�?�q�}:       ���	Ș;ND�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��Q�$       B+�M	�^"�D�A��*

Variable/sec��?��<:       ���	T�>�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�M�/$       B+�M	ta"�D�A��*

Variable/sec�?�{�w:       ���	a�A�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptMB��$       B+�M	�J"/D�A��*

Variable/sec���?���:       ���	�^D/D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt ��q$       B+�M	)^"zD�A��*

Variable/sec'~�?���:       ���	��DzD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptUivA$       B+�M	�X"�D�A��*

Variable/secƒ?K�:       ���	'�E�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptYE��$       B+�M	�:#D�A��*

Variable/secs1�?G�:       ���	�CCD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt$       B+�M	�e"[D�A��*

Variable/sec^Z�?��B:       ���	g�H[D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptg,g�$       B+�M	N"�D�A��*

Variable/secz~�?^��:       ���	�?:�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�+�$       B+�M	�S"�D�A��*

Variable/sec��?y+*�:       ���	2�@�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt^:�P$       B+�M	iU"<D�A��*

Variable/sec��?X2v�:       ���	t_<<D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt����$       B+�M	�O"�D�A��*

Variable/sec��?�>@:       ���	r�:�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�Y4$       B+�M	�#�D�A��*

Variable/secjy�?2&
�:       ���	28D�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptr�.�$       B+�M	�E"D�A��*

Variable/secⲄ?S�s�:       ���	S�FD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�;��$       B+�M	ׇ"hD�A��*

Variable/sec�o}?���:       ���	��_hD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpti�CT$       B+�M	X"�D�A��*

Variable/secz��?+η:       ���	pa9�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptt��m$       B+�M	�v"�D�A��*

Variable/sec~�?���K:       ���	N^;�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�$       B+�M	!Z"ID�A��*

Variable/sec�X�?0Uw:       ���	H=ID�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��M�$       B+�M		Q"�D�A��*

Variable/sec��?���:       ���	�$J�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpts;J$       B+�M	SX"�D�A��*

Variable/sec:~�?+kU:       ���	��>�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt˄1n$       B+�M	J"*D�A��*

Variable/sec~ɏ?2�A:       ���	 a*D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptqW�b$       B+�M	�W"uD�A��*

Variable/secj�?���:       ���	��LuD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�pA�$       B+�M	VI"�D�A��*

Variable/secƒ?97��:       ���	+�@�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�k��$       B+�M	=~"D�A��*

Variable/sec=�?%V�:       ���	:�@D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��J$       B+�M	5E#VD�A��*

Variable/sec�1�?2��6:       ���	D3?VD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptlaai$       B+�M	�#�D�A��*

Variable/sec��?)�9:       ���	�I;�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt� x/$       B+�M	'P"�D�Aɂ*

Variable/sec�?�?�+:       ���	I�I�D�Aʂ:+'logs/s2l_2017-05-06_13-29-45/model.ckptN^��$       B+�M	�U"7D�A��*

Variable/sec���?2��:       ���	͕A7D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��mF$       B+�M	7S"�D�A�*

Variable/sec�X�?>��:       ���	\�G�D�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckptOn$       B+�M	�"�D�A*

Variable/sec��?!�(�:       ���	��C�D�AÊ:+'logs/s2l_2017-05-06_13-29-45/model.ckptr��$       B+�M	�n"D�A��*

Variable/sec�~�?�u�:       ���	�ID�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�Ts$       B+�M	_C"cD�A�*

Variable/sec�~�?u�ѥ:       ���	��CcD�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M	�f"�D�A��*

Variable/sec!ɏ?u��:       ���	��L�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt{��"$       B+�M	�q"�D�A��*

Variable/sec�Œ?��s�:       ���	�>�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt;m�$       B+�M	�F"DD�A�*

Variable/secY�?L/��:       ���	m�9DD�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt4��$       B+�M	�B"�D�AÚ*

Variable/sec�X�?[O0�:       ���	u�9�D�AÚ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt<Y�$       B+�M	]N"�D�A��*

Variable/sec��?�8�:       ���	k>�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt)��b$       B+�M	�R"%D�A̟*

Variable/sec׃?��E':       ���	��B%D�A͟:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�F�$       B+�M	�y"pD�A��*

Variable/sec˭�?nK��:       ���	��DpD�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt}x�;$       B+�M	�]"�D�AƤ*

Variable/sec�ӆ?���<:       ���	��:�D�AǤ:+'logs/s2l_2017-05-06_13-29-45/model.ckptB�o�$       B+�M	�i"D�A��*

Variable/sec~_�?��&�:       ���	�?D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt[�ټ$       B+�M	h["QD�A�*

Variable/sec�X�?�%+t:       ���	�s<QD�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��!$       B+�M	dW"�D�A��*

Variable/secs��?���:       ���	)�9�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M	�P"�D�A��*

Variable/seciU�?��ϱ:       ���	��<�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptлI�$       B+�M	�U"2D�A��*

Variable/secb��?�?�B:       ���	f�D2D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptcJ<�$       B+�M	t]"}D�Aش*

Variable/sec�/�?�v�:       ���	T58}D�Aش:+'logs/s2l_2017-05-06_13-29-45/model.ckpt"o0u$       B+�M	CU"�D�A��*

Variable/secdU�?{�a:       ���	S[8�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt����$       B+�M	bN"D�A��*

Variable/sec*�?�0�:       ���	]�?D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt1s�$       B+�M	�S"^D�A�*

Variable/sec(3�?޺4�:       ���	��F^D�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�$       B+�M	nR"�D�Aſ*

Variable/sec73�?w�i:       ���	�;;�D�Aſ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt3�=�$       B+�M	�E"�D�A��*

Variable/sec�̌?�tW+:       ���	@^�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptKՍ#$       B+�M	�c"?D�A��*

Variable/sec��?����:       ���		k>?D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptk��Q$       B+�M	;R"�D�A��*

Variable/secm~�?���:       ���	^LI�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptm��5$       B+�M	�R"�D�A��*

Variable/sec�Œ?/i&:       ���	�G9�D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt٠2[$       B+�M	V�"  D�A��*

Variable/sec�Ē?��ǒ:       ���	Fz;  D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��K$       B+�M	�n"k D�A��*

Variable/sec�Y�?�gQy:       ���	��Ik D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptZ���$       B+�M	9{"� D�A��*

Variable/secHɏ?���:       ���	��<� D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�O~$       B+�M	�b"!D�A��*

Variable/sec��?~�`�:       ���	�6!D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt5�R$       B+�M	�"L!D�A��*

Variable/sec���?��XV:       ���	{:L!D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt����$       B+�M	�U"�!D�A��*

Variable/sec��?2��:       ���	t�n�!D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���!$       B+�M	�k"�!D�A��*

Variable/sec3�?�:       ���	��F�!D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�B��$       B+�M	-\"-"D�A��*

Variable/secz�?�2%h:       ���	R__-"D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M	�D"x"D�A��*

Variable/sec��?z}�:       ���	��=x"D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�S$       B+�M	_@"�"D�A��*

Variable/sec%K~?�+:       ���	��?�"D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt5�e$       B+�M	�_"#D�A��*

Variable/sec�i�?8��<:       ���	)�B#D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�໠$       B+�M	V"Y#D�A��*

Variable/sec��?��w8:       ���	p?FY#D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptK��`$       B+�M	�B"�#D�A��*

Variable/sec�U�?Q%:       ���	�V=�#D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptk��$       B+�M	M"�#D�A��*

Variable/sec_w�?xm��:       ���	B�C�#D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M	^h":$D�A��*

Variable/sec�/�?�7�:       ���	^.::$D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptC�$       B+�M	�h"�$D�A��*

Variable/sec��?z?�x:       ���	�<�$D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt� $       B+�M	�E"�$D�A��*

Variable/sec�w�?87;:       ���	q�:�$D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�v~�$       B+�M	�P"%D�A��*

Variable/sec(
�?���:       ���	��<%D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt^A^�$       B+�M	1A"f%D�A��*

Variable/sec^
�?Y���:       ���	��Bf%D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���[$       B+�M	�W"�%D�AĀ*

Variable/sec
�?
Ж�:       ���	 �B�%D�Aŀ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt]$       B+�M	XY"�%D�A��*

Variable/sec���?�^:       ���	��B�%D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	p"G&D�A��*

Variable/sec�Œ?��In:       ���	�=G&D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���B$       B+�M	�Z"�&D�A҈*

Variable/sect~�?-�?�:       ���	�H9�&D�Aӈ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	�R"�&D�A��*

Variable/secQ
�?����:       ���	J�<�&D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptr��$       B+�M	iQ"('D�A��*

Variable/sec<
�?G�u�:       ���	��@('D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M	N`"s'D�A��*

Variable/secYw�?���:       ���	A�;s'D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt6Uȑ$       B+�M	�J"�'D�Aٓ*

Variable/sec�/�?6�-:       ���	�m=�'D�Aٓ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�0i$       B+�M	�T"	(D�A��*

Variable/secOɏ?/ł�:       ���	��>	(D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt7��$       B+�M	 D"T(D�A��*

Variable/sec���?.��:       ���	$H@T(D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�\6[$       B+�M	p^"�(D�Aϛ*

Variable/seci6�?s� :       ���	�:�(D�Aϛ:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���$       B+�M	yY"�(D�A��*

Variable/secz=�?FWB^:       ���	�_<�(D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptb�L�$       B+�M	qU"5)D�AҠ*

Variable/secU�?n��:       ���	��D5)D�AҠ:+'logs/s2l_2017-05-06_13-29-45/model.ckptc�`�$       B+�M	�^"�)D�A��*

Variable/secfڀ?P�P:       ���	pB<�)D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��5$       B+�M	�@"�)D�A��*

Variable/sec��u?�L�:       ���	�?J�)D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptU�ˬ$       B+�M	yW"*D�A��*

Variable/sec��t?+�:       ���	�/A*D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt/=�$       B+�M	�U"a*D�A��*

Variable/sec$��?����:       ���	��<a*D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptM+,�$       B+�M	�R"�*D�A֬*

Variable/sec,\�??8:       ���	�tI�*D�A׬:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���e$       B+�M	<k"�*D�A��*

Variable/sec譇?Jx�3:       ���	_^Q�*D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�?�$       B+�M	IM"B+D�A�*

Variable/sec?e��:       ���	��JB+D�A�:+'logs/s2l_2017-05-06_13-29-45/model.ckpts�J~$       B+�M	UO"�+D�A��*

Variable/sec_ɏ?�)��:       ���	m�A�+D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt&xu"$       B+�M	�~"�+D�A��*

Variable/secɄ�?��p:       ���	! B�+D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���L$       B+�M	RG"#,D�A��*

Variable/sec&��?A�ݣ:       ���	�d;#,D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt*g��$       B+�M	�P"n,D�A��*

Variable/sec�9�?�	�:       ���	aSDn,D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpta-�J$       B+�M	�F"�,D�A־*

Variable/sec<\�?���:       ���	M�:�,D�A׾:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��H�$       B+�M	�K"-D�A��*

Variable/sec"\�?=���:       ���	�=-D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��$       B+�M	5`"O-D�A��*

Variable/sec��?ֶ�3:       ���	#L>O-D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��Ö$       B+�M	�Q"�-D�A��*

Variable/sec��?��V~:       ���	��?�-D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt#*$       B+�M	h\"�-D�A��*

Variable/sec�6�?��:       ���	��<�-D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptw�gJ$       B+�M	�O"0.D�A��*

Variable/sec�/�?�Q:       ���	G=H0.D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��_$       B+�M	��#{.D�A��*

Variable/secq�?ٹk:       ���	�)B{.D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt1�k	$       B+�M	�$�.D�A��*

Variable/secx�?�� :       ���	��<�.D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptޕ7�$       B+�M	E"/D�A��*

Variable/sec�Ь?Ӏa�:       ���	?</D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt���B$       B+�M	 r"\/D�A��*

Variable/sec�[�?,�:       ���	�<\/D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt��M$       B+�M	�#�/D�A��*

Variable/sec�~�?��b:       ���	�`<�/D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt�6��$       B+�M	�J"�/D�A��*

Variable/sec��?�:       ���	��F�/D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt� �$       B+�M	Gu"=0D�A��*

Variable/sec��?�/��:       ���	��F=0D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckptd�$       B+�M	M�"�0D�A��*

Variable/sec�C�?����:       ���	�TB�0D�A��:+'logs/s2l_2017-05-06_13-29-45/model.ckpt*)�u