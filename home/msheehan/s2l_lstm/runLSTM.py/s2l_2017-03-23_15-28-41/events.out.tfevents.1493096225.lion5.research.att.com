       ŁK"	  @Čľ?ÖAbrain.Event:2BXg     8ŘL	3QUČľ?ÖA"¤
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
b
inputPlaceholder*$
_output_shapes
:*
dtype0*
shape:
R
seq_lenPlaceholder*
_output_shapes	
:*
shape:*
dtype0
R
dec_lenPlaceholder*
_output_shapes	
:*
shape:*
dtype0
^
dec_targetsPlaceholder*
_output_shapes
:	*
shape:	*
dtype0
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
shape: *
dtype0*
shared_name 
˘
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
T0*
_class
loc:@Variable*
_output_shapes
: 
Ą
,input_embed/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
_class
loc:@input_embed*!
valueB"         

*input_embed/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
_class
loc:@input_embed*
valueB
 *
×Ł˝

*input_embed/Initializer/random_uniform/maxConst*
_class
loc:@input_embed*
valueB
 *
×Ł=*
dtype0*
_output_shapes
: 
ç
4input_embed/Initializer/random_uniform/RandomUniformRandomUniform,input_embed/Initializer/random_uniform/shape*#
_output_shapes
:*
dtype0*
seed2 *
_class
loc:@input_embed*
T0*

seed 
Ę
*input_embed/Initializer/random_uniform/subSub*input_embed/Initializer/random_uniform/max*input_embed/Initializer/random_uniform/min*
_class
loc:@input_embed*
_output_shapes
: *
T0
á
*input_embed/Initializer/random_uniform/mulMul4input_embed/Initializer/random_uniform/RandomUniform*input_embed/Initializer/random_uniform/sub*
_class
loc:@input_embed*#
_output_shapes
:*
T0
Ó
&input_embed/Initializer/random_uniformAdd*input_embed/Initializer/random_uniform/mul*input_embed/Initializer/random_uniform/min*#
_output_shapes
:*
_class
loc:@input_embed*
T0
Š
input_embed
VariableV2*
shape:*#
_output_shapes
:*
shared_name *
_class
loc:@input_embed*
dtype0*
	container 
Č
input_embed/AssignAssigninput_embed&input_embed/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@input_embed*
validate_shape(*#
_output_shapes
:
w
input_embed/readIdentityinput_embed*
T0*#
_output_shapes
:*
_class
loc:@input_embed
_
encoder/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 

encoder/conv1d/ExpandDims
ExpandDimsinputencoder/conv1d/ExpandDims/dim*(
_output_shapes
:*
T0*

Tdim0
a
encoder/conv1d/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 

encoder/conv1d/ExpandDims_1
ExpandDimsinput_embed/readencoder/conv1d/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:
ă
encoder/conv1d/Conv2DConv2Dencoder/conv1d/ExpandDimsencoder/conv1d/ExpandDims_1*
use_cudnn_on_gpu(*)
_output_shapes
:*
strides
*
data_formatNHWC*
T0*
paddingVALID

encoder/conv1d/SqueezeSqueezeencoder/conv1d/Conv2D*
squeeze_dims
*%
_output_shapes
:*
T0
Z
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         
]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
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
ů
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*
new_axis_mask *
_output_shapes
: *
shrink_axis_mask*
ellipsis_mask *

begin_mask *
end_mask 
Ź
)encoder/initial_state_0/Initializer/ConstConst**
_class 
loc:@encoder/initial_state_0*
valueB	*    *
_output_shapes
:	*
dtype0
š
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
ë
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
value	B :*
dtype0*
_output_shapes
: 
§
)encoder_1/initial_state_0_tiled/multiplesPackstrided_slice+encoder_1/initial_state_0_tiled/multiples/1*
_output_shapes
:*
N*

axis *
T0
ľ
encoder_1/initial_state_0_tiledTileencoder/initial_state_0/read)encoder_1/initial_state_0_tiled/multiples*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tmultiples0
Ź
)encoder/initial_state_1/Initializer/ConstConst**
_class 
loc:@encoder/initial_state_1*
valueB	*    *
dtype0*
_output_shapes
:	
š
encoder/initial_state_1
VariableV2**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	*
shape:	*
dtype0*
shared_name *
	container 
ë
encoder/initial_state_1/AssignAssignencoder/initial_state_1)encoder/initial_state_1/Initializer/Const*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	

encoder/initial_state_1/readIdentityencoder/initial_state_1*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1*
T0
m
+encoder_1/initial_state_1_tiled/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
§
)encoder_1/initial_state_1_tiled/multiplesPackstrided_slice+encoder_1/initial_state_1_tiled/multiples/1*

axis *
_output_shapes
:*
T0*
N
ľ
encoder_1/initial_state_1_tiledTileencoder/initial_state_1/read)encoder_1/initial_state_1_tiled/multiples*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tmultiples0
Ź
)encoder/initial_state_2/Initializer/ConstConst*
_output_shapes
:	*
dtype0**
_class 
loc:@encoder/initial_state_2*
valueB	*    
š
encoder/initial_state_2
VariableV2*
shape:	*
_output_shapes
:	*
shared_name **
_class 
loc:@encoder/initial_state_2*
dtype0*
	container 
ë
encoder/initial_state_2/AssignAssignencoder/initial_state_2)encoder/initial_state_2/Initializer/Const**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

encoder/initial_state_2/readIdentityencoder/initial_state_2*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2*
T0
m
+encoder_1/initial_state_2_tiled/multiples/1Const*
value	B :*
_output_shapes
: *
dtype0
§
)encoder_1/initial_state_2_tiled/multiplesPackstrided_slice+encoder_1/initial_state_2_tiled/multiples/1*
N*
T0*
_output_shapes
:*

axis 
ľ
encoder_1/initial_state_2_tiledTileencoder/initial_state_2/read)encoder_1/initial_state_2_tiled/multiples*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tmultiples0
Ź
)encoder/initial_state_3/Initializer/ConstConst*
dtype0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3*
valueB	*    
š
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
ë
encoder/initial_state_3/AssignAssignencoder/initial_state_3)encoder/initial_state_3/Initializer/Const*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	

encoder/initial_state_3/readIdentityencoder/initial_state_3**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	*
T0
m
+encoder_1/initial_state_3_tiled/multiples/1Const*
value	B :*
_output_shapes
: *
dtype0
§
)encoder_1/initial_state_3_tiled/multiplesPackstrided_slice+encoder_1/initial_state_3_tiled/multiples/1*
N*
T0*
_output_shapes
:*

axis 
ľ
encoder_1/initial_state_3_tiledTileencoder/initial_state_3/read)encoder_1/initial_state_3_tiled/multiples*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tmultiples0
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
encoder_1/sequence_lengthIdentityseq_len*
T0*
_output_shapes	
:
h
encoder_1/rnn/ShapeConst*!
valueB"         *
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
#encoder_1/rnn/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
m
#encoder_1/rnn/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
ż
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
%encoder_1/rnn/strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ç
encoder_1/rnn/strided_slice_1StridedSliceencoder_1/rnn/Shape#encoder_1/rnn/strided_slice_1/stack%encoder_1/rnn/strided_slice_1/stack_1%encoder_1/rnn/strided_slice_1/stack_2*
new_axis_mask *
shrink_axis_mask*
T0*
Index0*
end_mask *
_output_shapes
: *
ellipsis_mask *

begin_mask 
`
encoder_1/rnn/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
r
encoder_1/rnn/stackPackencoder_1/rnn/strided_slice*
T0*

axis *
N*
_output_shapes
:
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
encoder_1/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 

"encoder_1/rnn/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*J
valueAB? B9Expected shape for Tensor encoder_1/sequence_length:0 is 
s
"encoder_1/rnn/Assert/Assert/data_2Const*!
valueB B but saw shape: *
_output_shapes
: *
dtype0
Ě
encoder_1/rnn/Assert/AssertAssertencoder_1/rnn/All"encoder_1/rnn/Assert/Assert/data_0encoder_1/rnn/stack"encoder_1/rnn/Assert/Assert/data_2encoder_1/rnn/Shape_1*
T
2*
	summarize

encoder_1/rnn/CheckSeqLenIdentityencoder_1/sequence_length^encoder_1/rnn/Assert/Assert*
_output_shapes	
:*
T0
j
encoder_1/rnn/Shape_2Const*
_output_shapes
:*
dtype0*!
valueB"         
m
#encoder_1/rnn/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%encoder_1/rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
o
%encoder_1/rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
É
encoder_1/rnn/strided_slice_2StridedSliceencoder_1/rnn/Shape_2#encoder_1/rnn/strided_slice_2/stack%encoder_1/rnn/strided_slice_2/stack_1%encoder_1/rnn/strided_slice_2/stack_2*
ellipsis_mask *

begin_mask *
_output_shapes
: *
end_mask *
T0*
Index0*
shrink_axis_mask*
new_axis_mask 
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
valueB:*
dtype0*
_output_shapes
:
É
encoder_1/rnn/strided_slice_3StridedSliceencoder_1/rnn/Shape_2#encoder_1/rnn/strided_slice_3/stack%encoder_1/rnn/strided_slice_3/stack_1%encoder_1/rnn/strided_slice_3/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Z
encoder_1/rnn/stack_1/1Const*
value
B :*
_output_shapes
: *
dtype0
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
:˙˙˙˙˙˙˙˙˙
_
encoder_1/rnn/Const_1Const*
valueB: *
_output_shapes
:*
dtype0

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

encoder_1/rnn/MaxMaxencoder_1/rnn/CheckSeqLenencoder_1/rnn/Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
T
encoder_1/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
ô
encoder_1/rnn/TensorArrayTensorArrayV3encoder_1/rnn/strided_slice_2*
_output_shapes

::*
dtype0*
dynamic_size( *
clear_after_read(*9
tensor_array_name$"encoder_1/rnn/dynamic_rnn/output_0*
element_shape:
ő
encoder_1/rnn/TensorArray_1TensorArrayV3encoder_1/rnn/strided_slice_2*
dynamic_size( *
clear_after_read(*
_output_shapes

::*
element_shape:*
dtype0*8
tensor_array_name#!encoder_1/rnn/dynamic_rnn/input_0
{
&encoder_1/rnn/TensorArrayUnstack/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
~
4encoder_1/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

.encoder_1/rnn/TensorArrayUnstack/strided_sliceStridedSlice&encoder_1/rnn/TensorArrayUnstack/Shape4encoder_1/rnn/TensorArrayUnstack/strided_slice/stack6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_16encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
n
,encoder_1/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
n
,encoder_1/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
ě
&encoder_1/rnn/TensorArrayUnstack/rangeRange,encoder_1/rnn/TensorArrayUnstack/range/start.encoder_1/rnn/TensorArrayUnstack/strided_slice,encoder_1/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
Ş
Hencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3encoder_1/rnn/TensorArray_1&encoder_1/rnn/TensorArrayUnstack/rangeencoder_1/transposeencoder_1/rnn/TensorArray_1:1*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: 
ż
encoder_1/rnn/while/EnterEnterencoder_1/rnn/time*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
Ě
encoder_1/rnn/while/Enter_1Enterencoder_1/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
ŕ
encoder_1/rnn/while/Enter_2Enterencoder_1/initial_state_0_tiled*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
ŕ
encoder_1/rnn/while/Enter_3Enterencoder_1/initial_state_1_tiled*
parallel_iterations *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
ŕ
encoder_1/rnn/while/Enter_4Enterencoder_1/initial_state_2_tiled*
parallel_iterations *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
ŕ
encoder_1/rnn/while/Enter_5Enterencoder_1/initial_state_3_tiled*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/

encoder_1/rnn/while/MergeMergeencoder_1/rnn/while/Enter!encoder_1/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 

encoder_1/rnn/while/Merge_1Mergeencoder_1/rnn/while/Enter_1#encoder_1/rnn/while/NextIteration_1*
_output_shapes
:: *
N*
T0
¤
encoder_1/rnn/while/Merge_2Mergeencoder_1/rnn/while/Enter_2#encoder_1/rnn/while/NextIteration_2**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
N*
T0
¤
encoder_1/rnn/while/Merge_3Mergeencoder_1/rnn/while/Enter_3#encoder_1/rnn/while/NextIteration_3**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
N*
T0
¤
encoder_1/rnn/while/Merge_4Mergeencoder_1/rnn/while/Enter_4#encoder_1/rnn/while/NextIteration_4**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
N*
T0
¤
encoder_1/rnn/while/Merge_5Mergeencoder_1/rnn/while/Enter_5#encoder_1/rnn/while/NextIteration_5**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N
Ď
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
Ž
encoder_1/rnn/while/SwitchSwitchencoder_1/rnn/while/Mergeencoder_1/rnn/while/LoopCond*
T0*
_output_shapes
: : *,
_class"
 loc:@encoder_1/rnn/while/Merge
¸
encoder_1/rnn/while/Switch_1Switchencoder_1/rnn/while/Merge_1encoder_1/rnn/while/LoopCond*
T0*
_output_shapes

::*.
_class$
" loc:@encoder_1/rnn/while/Merge_1
Ř
encoder_1/rnn/while/Switch_2Switchencoder_1/rnn/while/Merge_2encoder_1/rnn/while/LoopCond*.
_class$
" loc:@encoder_1/rnn/while/Merge_2*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ř
encoder_1/rnn/while/Switch_3Switchencoder_1/rnn/while/Merge_3encoder_1/rnn/while/LoopCond*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@encoder_1/rnn/while/Merge_3
Ř
encoder_1/rnn/while/Switch_4Switchencoder_1/rnn/while/Merge_4encoder_1/rnn/while/LoopCond*
T0*.
_class$
" loc:@encoder_1/rnn/while/Merge_4*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ř
encoder_1/rnn/while/Switch_5Switchencoder_1/rnn/while/Merge_5encoder_1/rnn/while/LoopCond*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@encoder_1/rnn/while/Merge_5*
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
encoder_1/rnn/while/Identity_2Identityencoder_1/rnn/while/Switch_2:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
encoder_1/rnn/while/Identity_3Identityencoder_1/rnn/while/Switch_3:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
}
encoder_1/rnn/while/Identity_4Identityencoder_1/rnn/while/Switch_4:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
}
encoder_1/rnn/while/Identity_5Identityencoder_1/rnn/while/Switch_5:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

+encoder_1/rnn/while/TensorArrayReadV3/EnterEnterencoder_1/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
š
-encoder_1/rnn/while/TensorArrayReadV3/Enter_1EnterHencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
parallel_iterations *
is_constant(*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/

%encoder_1/rnn/while/TensorArrayReadV3TensorArrayReadV3+encoder_1/rnn/while/TensorArrayReadV3/Enterencoder_1/rnn/while/Identity-encoder_1/rnn/while/TensorArrayReadV3/Enter_1*.
_class$
" loc:@encoder_1/rnn/TensorArray_1* 
_output_shapes
:
*
dtype0
í
Tencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB"      
ß
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/minConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB
 *
×Ł˝*
_output_shapes
: *
dtype0
ß
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB
 *
×Ł=
Ü
\encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/RandomUniformRandomUniformTencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/shape*
seed2 *
T0*

seed *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:

ę
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/subSubRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/maxRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
ţ
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/mulMul\encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/RandomUniformRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:

đ
Nencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniformAddRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/mulRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:

ó
3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
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
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
ĺ
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
8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/readIdentity3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0* 
_output_shapes
:

Ş
Iencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axisConst^encoder_1/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
˘
Dencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concatConcatV2%encoder_1/rnn/while/TensorArrayReadV3encoder_1/rnn/while/Identity_3Iencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*

Tidx0*
T0*
N* 
_output_shapes
:

 
Jencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read*
parallel_iterations *
T0* 
_output_shapes
:
*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(
ą
Dencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMulMatMulDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concatJencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter*
transpose_b( * 
_output_shapes
:
*
transpose_a( *
T0
Ú
Dencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Initializer/ConstConst*
_output_shapes	
:*
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
valueB*    
ç
2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
VariableV2*
	container *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
shape:*
shared_name 
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
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/readIdentity2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0

Aencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/EnterEnter7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read*
_output_shapes	
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAddBiasAddDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMulAencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter* 
_output_shapes
:
*
T0*
data_formatNHWC
¤
Cencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dimConst^encoder_1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
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
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add/yConst^encoder_1/rnn/while/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?
á
7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/addAdd;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add/y*
T0* 
_output_shapes
:

Ş
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/SigmoidSigmoid7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add*
T0* 
_output_shapes
:

Ć
7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mulMul;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoidencoder_1/rnn/while/Identity_2*
T0* 
_output_shapes
:

Ž
=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1Sigmoid9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split*
T0* 
_output_shapes
:

¨
8encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/TanhTanh;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:1* 
_output_shapes
:
*
T0
ä
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1Mul=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_18encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh* 
_output_shapes
:
*
T0
ß
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1Add7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1*
T0* 
_output_shapes
:

°
=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2Sigmoid;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:3*
T0* 
_output_shapes
:

¨
:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1Tanh9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1* 
_output_shapes
:
*
T0
ć
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2Mul=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0* 
_output_shapes
:

í
Tencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB"      *
_output_shapes
:*
dtype0
ß
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB
 *
×Ł˝
ß
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/maxConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB
 *
×Ł=*
_output_shapes
: *
dtype0
Ü
\encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/RandomUniformRandomUniformTencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/shape*
seed2 *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*

seed * 
_output_shapes
:
*
T0
ę
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/subSubRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/maxRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/min*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0
ţ
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/mulMul\encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/RandomUniformRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/sub* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0
đ
Nencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniformAddRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/mulRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/min* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0
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
ĺ
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AssignAssign3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsNencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
validate_shape(* 
_output_shapes
:

¤
8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/readIdentity3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
T0
Ş
Iencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axisConst^encoder_1/rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :
ś
Dencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concatConcatV29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2encoder_1/rnn/while/Identity_5Iencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*

Tidx0*
T0*
N* 
_output_shapes
:

 
Jencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read*
is_constant(* 
_output_shapes
:
*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
ą
Dencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMulMatMulDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concatJencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
Ú
Dencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Initializer/ConstConst*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
valueB*    *
dtype0*
_output_shapes	
:
ç
2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
VariableV2*
	container *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
shared_name *
_output_shapes	
:*
shape:
Ó
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AssignAssign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesDencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Initializer/Const*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(

7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/readIdentity2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
_output_shapes	
:

Aencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/EnterEnter7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/read*
_output_shapes	
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAddBiasAddDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMulAencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter* 
_output_shapes
:
*
T0*
data_formatNHWC
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
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add/yConst^encoder_1/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  ?
á
7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/addAdd;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:29encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add/y* 
_output_shapes
:
*
T0
Ş
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/SigmoidSigmoid7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add* 
_output_shapes
:
*
T0
Ć
7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mulMul;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoidencoder_1/rnn/while/Identity_4*
T0* 
_output_shapes
:

Ž
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1Sigmoid9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split* 
_output_shapes
:
*
T0
¨
8encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/TanhTanh;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:1* 
_output_shapes
:
*
T0
ä
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1Mul=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_18encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0* 
_output_shapes
:

ß
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1Add7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1*
T0* 
_output_shapes
:

°
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2Sigmoid;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:3*
T0* 
_output_shapes
:

¨
:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1Tanh9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1*
T0* 
_output_shapes
:

ć
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2Mul=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1* 
_output_shapes
:
*
T0
Ř
&encoder_1/rnn/while/GreaterEqual/EnterEnterencoder_1/rnn/CheckSeqLen*
_output_shapes	
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

 encoder_1/rnn/while/GreaterEqualGreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:
Ů
 encoder_1/rnn/while/Select/EnterEnterencoder_1/rnn/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
Î
encoder_1/rnn/while/SelectSelect encoder_1/rnn/while/GreaterEqual encoder_1/rnn/while/Select/Enter9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0* 
_output_shapes
:


"encoder_1/rnn/while/GreaterEqual_1GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:
Đ
encoder_1/rnn/while/Select_1Select"encoder_1/rnn/while/GreaterEqual_1encoder_1/rnn/while/Identity_29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1* 
_output_shapes
:
*
T0

"encoder_1/rnn/while/GreaterEqual_2GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:
Đ
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
Đ
encoder_1/rnn/while/Select_3Select"encoder_1/rnn/while/GreaterEqual_3encoder_1/rnn/while/Identity_49encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1*
T0* 
_output_shapes
:


"encoder_1/rnn/while/GreaterEqual_4GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
_output_shapes	
:*
T0
Đ
encoder_1/rnn/while/Select_4Select"encoder_1/rnn/while/GreaterEqual_4encoder_1/rnn/while/Identity_59encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0* 
_output_shapes
:


=encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterencoder_1/rnn/TensorArray*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*,
_class"
 loc:@encoder_1/rnn/TensorArray
ľ
7encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3=encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterencoder_1/rnn/while/Identityencoder_1/rnn/while/Selectencoder_1/rnn/while/Identity_1*
T0*
_output_shapes
: *,
_class"
 loc:@encoder_1/rnn/TensorArray
z
encoder_1/rnn/while/add/yConst^encoder_1/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
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
#encoder_1/rnn/while/NextIteration_2NextIterationencoder_1/rnn/while/Select_1* 
_output_shapes
:
*
T0
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
#encoder_1/rnn/while/NextIteration_5NextIterationencoder_1/rnn/while/Select_4* 
_output_shapes
:
*
T0
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
:˙˙˙˙˙˙˙˙˙*
T0
s
encoder_1/rnn/while/Exit_3Exitencoder_1/rnn/while/Switch_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
encoder_1/rnn/while/Exit_4Exitencoder_1/rnn/while/Switch_4*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
encoder_1/rnn/while/Exit_5Exitencoder_1/rnn/while/Switch_5*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Â
0encoder_1/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3encoder_1/rnn/TensorArrayencoder_1/rnn/while/Exit_1*
_output_shapes
: *,
_class"
 loc:@encoder_1/rnn/TensorArray

*encoder_1/rnn/TensorArrayStack/range/startConst*
_output_shapes
: *
dtype0*
value	B : *,
_class"
 loc:@encoder_1/rnn/TensorArray

*encoder_1/rnn/TensorArrayStack/range/deltaConst*
value	B :*,
_class"
 loc:@encoder_1/rnn/TensorArray*
dtype0*
_output_shapes
: 

$encoder_1/rnn/TensorArrayStack/rangeRange*encoder_1/rnn/TensorArrayStack/range/start0encoder_1/rnn/TensorArrayStack/TensorArraySizeV3*encoder_1/rnn/TensorArrayStack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*,
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
encoder_1/rnn/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
ł
encoder_1/rnn/transpose	Transpose2encoder_1/rnn/TensorArrayStack/TensorArrayGatherV3encoder_1/rnn/transpose/perm*
Tperm0*%
_output_shapes
:*
T0
Ż
6output_projection/W/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*&
_class
loc:@output_projection/W*
valueB"      
˘
5output_projection/W/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*&
_class
loc:@output_projection/W*
valueB
 *    
¤
7output_projection/W/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *&
_class
loc:@output_projection/W*
valueB
 *ÍĚĚ=

@output_projection/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6output_projection/W/Initializer/truncated_normal/shape*
T0*
_output_shapes
:	*

seed *&
_class
loc:@output_projection/W*
dtype0*
seed2 

4output_projection/W/Initializer/truncated_normal/mulMul@output_projection/W/Initializer/truncated_normal/TruncatedNormal7output_projection/W/Initializer/truncated_normal/stddev*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0
ö
0output_projection/W/Initializer/truncated_normalAdd4output_projection/W/Initializer/truncated_normal/mul5output_projection/W/Initializer/truncated_normal/mean*
T0*&
_class
loc:@output_projection/W*
_output_shapes
:	
ą
output_projection/W
VariableV2*
	container *
shared_name *
dtype0*
shape:	*
_output_shapes
:	*&
_class
loc:@output_projection/W
ć
output_projection/W/AssignAssignoutput_projection/W0output_projection/W/Initializer/truncated_normal*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*&
_class
loc:@output_projection/W

output_projection/W/readIdentityoutput_projection/W*
_output_shapes
:	*&
_class
loc:@output_projection/W*
T0

%output_projection/b/Initializer/ConstConst*&
_class
loc:@output_projection/b*
valueB*ÍĚĚ=*
dtype0*
_output_shapes
:
§
output_projection/b
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_output_shapes
:*&
_class
loc:@output_projection/b
Ö
output_projection/b/AssignAssignoutput_projection/b%output_projection/b/Initializer/Const*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

output_projection/b/readIdentityoutput_projection/b*
T0*&
_class
loc:@output_projection/b*
_output_shapes
:
ş
"output_projection/xw_plus_b/MatMulMatMulencoder_1/rnn/while/Exit_4output_projection/W/read*
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0
­
output_projection/xw_plus_bBiasAdd"output_projection/xw_plus_b/MatMuloutput_projection/b/read*
data_formatNHWC*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
output_projection/SoftmaxSoftmaxoutput_projection/xw_plus_b*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
d
"output_projection/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

output_projection/ArgMaxArgMaxoutput_projection/xw_plus_b"output_projection/ArgMax/dimension*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tidx0
o

loss/ConstConst*
dtype0*
_output_shapes
:*1
value(B&"  ?žz?`ĺp?bX?¤p}?  ?mç{>
j
loss/MulMuloutput_projection/xw_plus_b
loss/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
W
	loss/CastCastdec_targets*

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
loss/logistic_loss/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
p
loss/logistic_loss/sub_1Subloss/logistic_loss/sub_1/x	loss/Cast*
T0*
_output_shapes
:	
m
loss/logistic_loss/mul_1Mulloss/logistic_loss/sub_1loss/Mul*
_output_shapes
:	*
T0
Y
loss/logistic_loss/AbsAbsloss/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
loss/logistic_loss/NegNegloss/logistic_loss/Abs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
loss/logistic_loss/ExpExploss/logistic_loss/Neg*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
loss/logistic_loss/Log1pLog1ploss/logistic_loss/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
loss/logistic_loss/Neg_1Negloss/Mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
k
loss/logistic_loss/ReluReluloss/logistic_loss/Neg_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

loss/logistic_loss/add_1Addloss/logistic_loss/Log1ploss/logistic_loss/Relu*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
accuracy/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
s
accuracy/ArgMaxArgMaxdec_targetsaccuracy/ArgMax/dimension*
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
accuracy/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
v
accuracy/accuracyMeanaccuracy/Castaccuracy/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
`
learning_rate/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *ˇQ8
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
learning_rate/Cast_1Castlearning_rate/Cast_1/x*
_output_shapes
: *

DstT0*

SrcT0
[
learning_rate/Cast_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *Âu?
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
learning_rateMullearning_rate/learning_ratelearning_rate/Pow*
_output_shapes
: *
T0
`
gradients/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
T
gradients/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
b
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
:	
S
gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
¸
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
š
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
gradients/b_countConst*
_output_shapes
: *
dtype0*
value	B :
Ä
gradients/b_count_1Entergradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
_output_shapes
: : *
N*
T0
Ë
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
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 

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
'gradients/loss/logistic_loss_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      
z
)gradients/loss/logistic_loss_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      
á
7gradients/loss/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/loss/logistic_loss_grad/Shape)gradients/loss/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ľ
%gradients/loss/logistic_loss_grad/SumSumgradients/Fill7gradients/loss/logistic_loss_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ź
)gradients/loss/logistic_loss_grad/ReshapeReshape%gradients/loss/logistic_loss_grad/Sum'gradients/loss/logistic_loss_grad/Shape*
T0*
_output_shapes
:	*
Tshape0
š
'gradients/loss/logistic_loss_grad/Sum_1Sumgradients/Fill9gradients/loss/logistic_loss_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Â
+gradients/loss/logistic_loss_grad/Reshape_1Reshape'gradients/loss/logistic_loss_grad/Sum_1)gradients/loss/logistic_loss_grad/Shape_1*
_output_shapes
:	*
Tshape0*
T0
~
-gradients/loss/logistic_loss/mul_1_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
w
/gradients/loss/logistic_loss/mul_1_grad/Shape_1Shapeloss/Mul*
T0*
_output_shapes
:*
out_type0
ó
=gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/mul_1_grad/Shape/gradients/loss/logistic_loss/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

+gradients/loss/logistic_loss/mul_1_grad/mulMul)gradients/loss/logistic_loss_grad/Reshapeloss/Mul*
T0*
_output_shapes
:	
Ţ
+gradients/loss/logistic_loss/mul_1_grad/SumSum+gradients/loss/logistic_loss/mul_1_grad/mul=gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Î
/gradients/loss/logistic_loss/mul_1_grad/ReshapeReshape+gradients/loss/logistic_loss/mul_1_grad/Sum-gradients/loss/logistic_loss/mul_1_grad/Shape*
_output_shapes
:	*
Tshape0*
T0
Ł
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
1gradients/loss/logistic_loss/mul_1_grad/Reshape_1Reshape-gradients/loss/logistic_loss/mul_1_grad/Sum_1/gradients/loss/logistic_loss/mul_1_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
~
-gradients/loss/logistic_loss/mul_2_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      

/gradients/loss/logistic_loss/mul_2_grad/Shape_1Shapeloss/logistic_loss/add_1*
T0*
_output_shapes
:*
out_type0
ó
=gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/mul_2_grad/Shape/gradients/loss/logistic_loss/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ł
+gradients/loss/logistic_loss/mul_2_grad/mulMul+gradients/loss/logistic_loss_grad/Reshape_1loss/logistic_loss/add_1*
T0*
_output_shapes
:	
Ţ
+gradients/loss/logistic_loss/mul_2_grad/SumSum+gradients/loss/logistic_loss/mul_2_grad/mul=gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Î
/gradients/loss/logistic_loss/mul_2_grad/ReshapeReshape+gradients/loss/logistic_loss/mul_2_grad/Sum-gradients/loss/logistic_loss/mul_2_grad/Shape*
T0*
_output_shapes
:	*
Tshape0
Ł
-gradients/loss/logistic_loss/mul_2_grad/mul_1Mulloss/logistic_loss/add+gradients/loss/logistic_loss_grad/Reshape_1*
T0*
_output_shapes
:	
ä
-gradients/loss/logistic_loss/mul_2_grad/Sum_1Sum-gradients/loss/logistic_loss/mul_2_grad/mul_1?gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ü
1gradients/loss/logistic_loss/mul_2_grad/Reshape_1Reshape-gradients/loss/logistic_loss/mul_2_grad/Sum_1/gradients/loss/logistic_loss/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

-gradients/loss/logistic_loss/add_1_grad/ShapeShapeloss/logistic_loss/Log1p*
T0*
out_type0*
_output_shapes
:

/gradients/loss/logistic_loss/add_1_grad/Shape_1Shapeloss/logistic_loss/Relu*
T0*
_output_shapes
:*
out_type0
ó
=gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/add_1_grad/Shape/gradients/loss/logistic_loss/add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ä
+gradients/loss/logistic_loss/add_1_grad/SumSum1gradients/loss/logistic_loss/mul_2_grad/Reshape_1=gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ö
/gradients/loss/logistic_loss/add_1_grad/ReshapeReshape+gradients/loss/logistic_loss/add_1_grad/Sum-gradients/loss/logistic_loss/add_1_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
¤
-gradients/loss/logistic_loss/Log1p_grad/add/xConst0^gradients/loss/logistic_loss/add_1_grad/Reshape*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ť
+gradients/loss/logistic_loss/Log1p_grad/addAdd-gradients/loss/logistic_loss/Log1p_grad/add/xloss/logistic_loss/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

2gradients/loss/logistic_loss/Log1p_grad/Reciprocal
Reciprocal+gradients/loss/logistic_loss/Log1p_grad/add*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
É
+gradients/loss/logistic_loss/Log1p_grad/mulMul/gradients/loss/logistic_loss/add_1_grad/Reshape2gradients/loss/logistic_loss/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
/gradients/loss/logistic_loss/Relu_grad/ReluGradReluGrad1gradients/loss/logistic_loss/add_1_grad/Reshape_1loss/logistic_loss/Relu*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
)gradients/loss/logistic_loss/Exp_grad/mulMul+gradients/loss/logistic_loss/Log1p_grad/mulloss/logistic_loss/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

+gradients/loss/logistic_loss/Neg_1_grad/NegNeg/gradients/loss/logistic_loss/Relu_grad/ReluGrad*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

)gradients/loss/logistic_loss/Neg_grad/NegNeg)gradients/loss/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
*gradients/loss/logistic_loss/Abs_grad/SignSignloss/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
)gradients/loss/logistic_loss/Abs_grad/mulMul)gradients/loss/logistic_loss/Neg_grad/Neg*gradients/loss/logistic_loss/Abs_grad/Sign*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˘
gradients/AddNAddN1gradients/loss/logistic_loss/mul_1_grad/Reshape_1+gradients/loss/logistic_loss/Neg_1_grad/Neg)gradients/loss/logistic_loss/Abs_grad/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N*D
_class:
86loc:@gradients/loss/logistic_loss/mul_1_grad/Reshape_1*
T0
x
gradients/loss/Mul_grad/ShapeShapeoutput_projection/xw_plus_b*
T0*
out_type0*
_output_shapes
:
i
gradients/loss/Mul_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
Ă
-gradients/loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/Mul_grad/Shapegradients/loss/Mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
p
gradients/loss/Mul_grad/mulMulgradients/AddN
loss/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
gradients/loss/Mul_grad/SumSumgradients/loss/Mul_grad/mul-gradients/loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ś
gradients/loss/Mul_grad/ReshapeReshapegradients/loss/Mul_grad/Sumgradients/loss/Mul_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/loss/Mul_grad/mul_1Muloutput_projection/xw_plus_bgradients/AddN*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients/loss/Mul_grad/Sum_1Sumgradients/loss/Mul_grad/mul_1/gradients/loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

!gradients/loss/Mul_grad/Reshape_1Reshapegradients/loss/Mul_grad/Sum_1gradients/loss/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
˘
6gradients/output_projection/xw_plus_b_grad/BiasAddGradBiasAddGradgradients/loss/Mul_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
Ö
8gradients/output_projection/xw_plus_b/MatMul_grad/MatMulMatMulgradients/loss/Mul_grad/Reshapeoutput_projection/W/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ń
:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1MatMulencoder_1/rnn/while/Exit_4gradients/loss/Mul_grad/Reshape*
transpose_b( *
_output_shapes
:	*
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
:˙˙˙˙˙˙˙˙˙*
T0
r
gradients/zeros_like_2	ZerosLikeencoder_1/rnn/while/Exit_3*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
gradients/zeros_like_3	ZerosLikeencoder_1/rnn/while/Exit_5*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

0gradients/encoder_1/rnn/while/Exit_4_grad/b_exitEnter8gradients/output_projection/xw_plus_b/MatMul_grad/MatMul*
is_constant( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
ä
0gradients/encoder_1/rnn/while/Exit_1_grad/b_exitEntergradients/zeros_like*
parallel_iterations *
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
ö
0gradients/encoder_1/rnn/while/Exit_2_grad/b_exitEntergradients/zeros_like_1*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
ö
0gradients/encoder_1/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_like_2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
ö
0gradients/encoder_1/rnn/while/Exit_5_grad/b_exitEntergradients/zeros_like_3*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
ę
4gradients/encoder_1/rnn/while/Switch_4_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_4_grad/b_exit;gradients/encoder_1/rnn/while/Switch_4_grad_1/NextIteration**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N
ę
4gradients/encoder_1/rnn/while/Switch_2_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_2_grad/b_exit;gradients/encoder_1/rnn/while/Switch_2_grad_1/NextIteration*
N*
T0**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
ę
4gradients/encoder_1/rnn/while/Switch_3_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_3_grad/b_exit;gradients/encoder_1/rnn/while/Switch_3_grad_1/NextIteration**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
N*
T0
ę
4gradients/encoder_1/rnn/while/Switch_5_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_5_grad/b_exit;gradients/encoder_1/rnn/while/Switch_5_grad_1/NextIteration**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N

1gradients/encoder_1/rnn/while/Merge_4_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_4_grad/b_switch*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙:
*
T0

1gradients/encoder_1/rnn/while/Merge_2_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙:
*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_2_grad/b_switch*
T0

1gradients/encoder_1/rnn/while/Merge_3_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_3_grad/b_switch*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙:
*
T0

1gradients/encoder_1/rnn/while/Merge_5_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_5_grad/b_switchgradients/b_count_2*
T0*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_5_grad/b_switch*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙:


/gradients/encoder_1/rnn/while/Enter_4_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_4_grad/Switch*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/encoder_1/rnn/while/Enter_2_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_2_grad/Switch*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

/gradients/encoder_1/rnn/while/Enter_3_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_3_grad/Switch*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/encoder_1/rnn/while/Enter_5_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_5_grad/Switch*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

4gradients/encoder_1/initial_state_2_tiled_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      
Ű
4gradients/encoder_1/initial_state_2_tiled_grad/stackPack)encoder_1/initial_state_2_tiled/multiples4gradients/encoder_1/initial_state_2_tiled_grad/Shape*
_output_shapes

:*
N*

axis *
T0

=gradients/encoder_1/initial_state_2_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_2_tiled_grad/stack*
_output_shapes
: *
T0

>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub/yConst*
value	B :*
_output_shapes
: *
dtype0
ă
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
value	B :*
_output_shapes
: *
dtype0
ş
>gradients/encoder_1/initial_state_2_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_2_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/delta*
_output_shapes
:*

Tidx0
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
˙˙˙˙˙˙˙˙˙
ě
6gradients/encoder_1/initial_state_2_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_2_tiled_grad/transpose<gradients/encoder_1/initial_state_2_tiled_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
u
3gradients/encoder_1/initial_state_2_tiled_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
|
:gradients/encoder_1/initial_state_2_tiled_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
|
:gradients/encoder_1/initial_state_2_tiled_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

4gradients/encoder_1/initial_state_2_tiled_grad/rangeRange:gradients/encoder_1/initial_state_2_tiled_grad/range/start3gradients/encoder_1/initial_state_2_tiled_grad/Size:gradients/encoder_1/initial_state_2_tiled_grad/range/delta*
_output_shapes
:*

Tidx0

8gradients/encoder_1/initial_state_2_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_4_grad/Exit6gradients/encoder_1/initial_state_2_tiled_grad/Reshape*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tshape0
đ
2gradients/encoder_1/initial_state_2_tiled_grad/SumSum8gradients/encoder_1/initial_state_2_tiled_grad/Reshape_14gradients/encoder_1/initial_state_2_tiled_grad/range*
_output_shapes
:	*
T0*
	keep_dims( *

Tidx0
ˇ
<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_accStack*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*

stack_name *
_output_shapes
:
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
Hgradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
T0*
is_constant(

?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/b_syncControlTrigger@^gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopa^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopd^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopa^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopd^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopb^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop
ˇ
6gradients/encoder_1/rnn/while/Select_3_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_accStack*
	elem_type0
*
_output_shapes
:*

stack_name *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3
Ĺ
;gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc*
is_constant(*
T0*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
§
<gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_3^gradients/Add*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
_output_shapes
:*
swap_memory( *
T0

Ř
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
4gradients/encoder_1/initial_state_0_tiled_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ű
4gradients/encoder_1/initial_state_0_tiled_grad/stackPack)encoder_1/initial_state_0_tiled/multiples4gradients/encoder_1/initial_state_0_tiled_grad/Shape*
T0*

axis *
N*
_output_shapes

:
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
ă
<gradients/encoder_1/initial_state_0_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_0_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub/y*
T0*
_output_shapes
: 
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
ş
>gradients/encoder_1/initial_state_0_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_0_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/delta*
_output_shapes
:*

Tidx0
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
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
ě
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
8gradients/encoder_1/initial_state_0_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_2_grad/Exit6gradients/encoder_1/initial_state_0_tiled_grad/Reshape*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
đ
2gradients/encoder_1/initial_state_0_tiled_grad/SumSum8gradients/encoder_1/initial_state_0_tiled_grad/Reshape_14gradients/encoder_1/initial_state_0_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	
ˇ
<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *1
_class'
%#loc:@encoder_1/rnn/while/Identity_2
É
?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
§
@gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_2^gradients/Add*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*
_output_shapes
:*
swap_memory( 
Ü
Hgradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2

?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
6gradients/encoder_1/rnn/while/Select_1_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ˇ
8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*

stack_name *
_output_shapes
:
Ĺ
;gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1
§
<gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_1^gradients/Add*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
_output_shapes
:*
swap_memory( *
T0

Ř
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
4gradients/encoder_1/rnn/while/Select_1_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_1_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_2_grad/Switch:1* 
_output_shapes
:
*
T0

4gradients/encoder_1/initial_state_1_tiled_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      
Ű
4gradients/encoder_1/initial_state_1_tiled_grad/stackPack)encoder_1/initial_state_1_tiled/multiples4gradients/encoder_1/initial_state_1_tiled_grad/Shape*
T0*

axis *
N*
_output_shapes

:

=gradients/encoder_1/initial_state_1_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_1_tiled_grad/stack*
_output_shapes
: *
T0

>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub/yConst*
dtype0*
_output_shapes
: *
value	B :
ă
<gradients/encoder_1/initial_state_1_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_1_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub/y*
_output_shapes
: *
T0

Dgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/startConst*
value	B : *
_output_shapes
: *
dtype0

Dgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ş
>gradients/encoder_1/initial_state_1_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_1_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/delta*
_output_shapes
:*

Tidx0
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
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
ě
6gradients/encoder_1/initial_state_1_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_1_tiled_grad/transpose<gradients/encoder_1/initial_state_1_tiled_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
u
3gradients/encoder_1/initial_state_1_tiled_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
|
:gradients/encoder_1/initial_state_1_tiled_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
|
:gradients/encoder_1/initial_state_1_tiled_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

4gradients/encoder_1/initial_state_1_tiled_grad/rangeRange:gradients/encoder_1/initial_state_1_tiled_grad/range/start3gradients/encoder_1/initial_state_1_tiled_grad/Size:gradients/encoder_1/initial_state_1_tiled_grad/range/delta*
_output_shapes
:*

Tidx0

8gradients/encoder_1/initial_state_1_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_3_grad/Exit6gradients/encoder_1/initial_state_1_tiled_grad/Reshape*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
đ
2gradients/encoder_1/initial_state_1_tiled_grad/SumSum8gradients/encoder_1/initial_state_1_tiled_grad/Reshape_14gradients/encoder_1/initial_state_1_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	
ˇ
<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3
É
?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
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
Hgradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/

?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3
ˇ
6gradients/encoder_1/rnn/while/Select_2_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*

stack_name *
_output_shapes
:
Ĺ
;gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
T0*
is_constant(
§
<gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_2^gradients/Add*
_output_shapes
:*
swap_memory( *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
T0

Ř
Dgradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
T0*
is_constant(
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
4gradients/encoder_1/rnn/while/Select_2_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_2_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_3_grad/Switch:1*
T0* 
_output_shapes
:


4gradients/encoder_1/initial_state_3_tiled_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ű
4gradients/encoder_1/initial_state_3_tiled_grad/stackPack)encoder_1/initial_state_3_tiled/multiples4gradients/encoder_1/initial_state_3_tiled_grad/Shape*
_output_shapes

:*
N*

axis *
T0

=gradients/encoder_1/initial_state_3_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_3_tiled_grad/stack*
T0*
_output_shapes
: 

>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
ă
<gradients/encoder_1/initial_state_3_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_3_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub/y*
T0*
_output_shapes
: 

Dgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 

Dgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
ş
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
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
ě
6gradients/encoder_1/initial_state_3_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_3_tiled_grad/transpose<gradients/encoder_1/initial_state_3_tiled_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
u
3gradients/encoder_1/initial_state_3_tiled_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
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

4gradients/encoder_1/initial_state_3_tiled_grad/rangeRange:gradients/encoder_1/initial_state_3_tiled_grad/range/start3gradients/encoder_1/initial_state_3_tiled_grad/Size:gradients/encoder_1/initial_state_3_tiled_grad/range/delta*
_output_shapes
:*

Tidx0

8gradients/encoder_1/initial_state_3_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_5_grad/Exit6gradients/encoder_1/initial_state_3_tiled_grad/Reshape*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tshape0
đ
2gradients/encoder_1/initial_state_3_tiled_grad/SumSum8gradients/encoder_1/initial_state_3_tiled_grad/Reshape_14gradients/encoder_1/initial_state_3_tiled_grad/range*
_output_shapes
:	*
T0*
	keep_dims( *

Tidx0
ˇ
<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_accStack*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*

stack_name *
_output_shapes
:
É
?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
§
@gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_5^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *1
_class'
%#loc:@encoder_1/rnn/while/Identity_5
Ü
Hgradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*
T0

?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5
ˇ
6gradients/encoder_1/rnn/while/Select_4_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ˇ
8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_accStack*
	elem_type0
*

stack_name *
_output_shapes
:*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4
Ĺ
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
<gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_4^gradients/Add*
T0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
_output_shapes
:*
swap_memory( 
Ř
Dgradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
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
Ż
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
ą
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
é
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_accStack*
	elem_type0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
_output_shapes
:*

stack_name 
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
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1
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
Á
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
˛
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape*
T0* 
_output_shapes
:
*
Tshape0
î
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc*
is_constant(*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2^gradients/Add*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
_output_shapes
:*
swap_memory( *
T0
Ť
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2
Ú
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2* 
_output_shapes
:


Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop4gradients/encoder_1/rnn/while/Select_4_grad/Select_1*
T0* 
_output_shapes
:

Ç
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¸
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*
T0*
Tshape0* 
_output_shapes
:

˝
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape*
T0* 
_output_shapes
:

´
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1* 
_output_shapes
:
*
T0

gradients/AddN_1AddN4gradients/encoder_1/rnn/while/Select_3_grad/Select_1Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGrad*
N*
T0* 
_output_shapes
:
*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_3_grad/Select_1
Ż
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
ą
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/SumSumgradients/AddN_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
˛
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0

Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¸
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1*
T0*
Tshape0* 
_output_shapes
:

­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
Ź
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1Shapeencoder_1/rnn/while/Identity_4*
out_type0*
_output_shapes
:*
T0

bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1
Ĺ
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
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1^gradients/Add*
T0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*
swap_memory( 
Ř
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
ç
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shapeegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mulMulPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop* 
_output_shapes
:
*
T0
ť
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumSumJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ź
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape* 
_output_shapes
:
*
Tshape0*
T0
ę
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc*
T0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnter;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid^gradients/Add*
_output_shapes
:*
swap_memory( *N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
T0
Ľ
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
T0
Ô
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
Á
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ń
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
Ż
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
ą
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ç
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnter8encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh^gradients/Add*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
_output_shapes
:*
swap_memory( *
T0
˘
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0*
is_constant(
Ń
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh* 
_output_shapes
:

Š
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mulMulRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop*
T0* 
_output_shapes
:

Á
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
˛
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0
î
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_accStack*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:
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
T0*
_output_shapes
:*
swap_memory( *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1
Ť
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
Ú
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1* 
_output_shapes
:

­
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1* 
_output_shapes
:
*
T0
Ç
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¸
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1* 
_output_shapes
:
*
Tshape0*
T0
ˇ
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
˝
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape*
T0* 
_output_shapes
:

˛
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1*
T0* 
_output_shapes
:

­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
Ą
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
: *
valueB 
Đ
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ç
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumSumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ź
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape* 
_output_shapes
:
*
Tshape0*
T0
Ë
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1SumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¨
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0

;gradients/encoder_1/rnn/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_2* 
_output_shapes
:
*
T0
ő
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim
 
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc*
T0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
Ł
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPush	StackPushXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnterCencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim
ł
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
Ř
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPopStackPopagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop/RefEnter^gradients/Sub*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
_output_shapes
: 
Ë
Ogradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concatConcatV2Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop*
N*

Tidx0*
T0* 
_output_shapes
:

ó
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
_output_shapes	
:*
data_formatNHWC*
T0
Ŕ
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
	elem_type0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*

stack_name *
_output_shapes
:
ť
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat
ż
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat^gradients/Add*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
_output_shapes
:*
swap_memory( *
T0
Î
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
ý
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat
ď
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1MatMulegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
Ľ
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
_output_shapes	
:*
dtype0
Ń
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc*
_output_shapes	
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
Ě
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
T0*
_output_shapes
	:: 
ý
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
::
´
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/AddAddYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch:1Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
ë
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
ß
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:*
T0
Ş
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/RankConst^gradients/Sub*
dtype0*
_output_shapes
: *
value	B :

]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis
ś
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
T0
ż
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPush	StackPush`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnterIencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis^gradients/Add*
_output_shapes
:*
swap_memory( *\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
T0
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
î
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopStackPopigradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnter^gradients/Sub*
	elem_type0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
_output_shapes
: 
Ŕ
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/modFloorMod`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
ş
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
š
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Shape_1Shapeencoder_1/rnn/while/Identity_5*
_output_shapes
:*
out_type0*
T0
ö
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_accStack*
	elem_type0*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:*

stack_name 
Ź
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
Ľ
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPush	StackPushcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2^gradients/Add*
T0*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:*
swap_memory( 
ż
lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
T0
î
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopStackPoplgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnter^gradients/Sub*
	elem_type0*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2* 
_output_shapes
:

Î
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeNShapeNcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop*
out_type0* 
_output_shapes
::*
T0*
N
Ž
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffsetConcatOffsetWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/modZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
´
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/SliceSliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffsetZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN*
Index0*
T0* 
_output_shapes
:

Â
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice_1SliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMulbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffset:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Index0*
T0
¸
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
ě
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2Mergeagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIteration*"
_output_shapes
:
: *
T0*
N

`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/SwitchSwitchagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_output_shapes
:
:
*
T0
Ń
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/AddAddbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:


ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIterationNextIteration]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
*
T0
ö
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3Exit`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:

Ś
gradients/AddN_3AddN4gradients/encoder_1/rnn/while/Select_2_grad/Select_1Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_2_grad/Select_1* 
_output_shapes
:
*
T0*
N
Ż
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
ą
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
é
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_accStack*
	elem_type0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
_output_shapes
:*

stack_name 

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc*
T0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnter:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1^gradients/Add*
_output_shapes
:*
swap_memory( *M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0
¤
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc*
T0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
Ó
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1* 
_output_shapes
:

ç
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mulMulgradients/AddN_3Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop*
T0* 
_output_shapes
:

Á
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
˛
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0
î
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc*
is_constant(*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2
Ť
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
T0
Ú
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2
ë
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopgradients/AddN_3*
T0* 
_output_shapes
:

Ç
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¸
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1* 
_output_shapes
:
*
Tshape0*
T0
¤
gradients/AddN_4AddN2gradients/encoder_1/rnn/while/Select_4_grad/Select[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice_1* 
_output_shapes
:
*
N*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_4_grad/Select*
T0
˝
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape*
T0* 
_output_shapes
:

´
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1*
T0* 
_output_shapes
:


;gradients/encoder_1/rnn/while/Switch_5_grad_1/NextIterationNextIterationgradients/AddN_4* 
_output_shapes
:
*
T0

gradients/AddN_5AddN4gradients/encoder_1/rnn/while/Select_1_grad/Select_1Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGrad*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_1_grad/Select_1* 
_output_shapes
:
*
T0*
N
Ż
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
ą
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/SumSumgradients/AddN_5^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
˛
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
¸
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1* 
_output_shapes
:
*
Tshape0*
T0
­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ShapeConst^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
Ź
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
Ĺ
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
Ó
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1^gradients/Add*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*
swap_memory( *
T0
Ř
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1

egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
:*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1
ç
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shapeegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mulMulPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop*
T0* 
_output_shapes
:

ť
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumSumJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ź
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape*
T0*
Tshape0* 
_output_shapes
:

ę
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_accStack*
	elem_type0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*

stack_name *
_output_shapes
:

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc*
is_constant(*
T0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnter;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid^gradients/Add*
_output_shapes
:*
swap_memory( *N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
T0
Ľ
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc*
T0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
Ô
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid* 
_output_shapes
:

§
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1MulUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape*
T0* 
_output_shapes
:

Á
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ń
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ż
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
ą
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ç
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_accStack*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
_output_shapes
:*

stack_name 

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
T0*
is_constant(

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnter8encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh^gradients/Add*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
_output_shapes
:*
swap_memory( *
T0
˘
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
T0*
is_constant(
Ń
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh* 
_output_shapes
:

Š
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mulMulRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop* 
_output_shapes
:
*
T0
Á
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
˛
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape*
T0* 
_output_shapes
:
*
Tshape0
î
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_accStack*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1^gradients/Add*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
_output_shapes
:*
swap_memory( *
T0
Ť
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1
Ú
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
Ç
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¸
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1* 
_output_shapes
:
*
Tshape0*
T0
ˇ
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape* 
_output_shapes
:
*
T0

gradients/AddN_6AddN2gradients/encoder_1/rnn/while/Select_1_grad/SelectPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1*
N*
T0* 
_output_shapes
:
*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_1_grad/Select
˝
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape*
T0* 
_output_shapes
:

˛
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
Ą
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
: *
valueB 
Đ
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ç
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumSumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ź
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0
Ë
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1SumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¨
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0

;gradients/encoder_1/rnn/while/Switch_2_grad_1/NextIterationNextIterationgradients/AddN_6*
T0* 
_output_shapes
:

ő
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_accStack*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
_output_shapes
:*

stack_name 
 
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
T0
Ł
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPush	StackPushXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnterCencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim
ł
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
Ř
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPopStackPopagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
: *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim
Ë
Ogradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concatConcatV2Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop*
N*

Tidx0*
T0* 
_output_shapes
:

ó
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:
Ŕ
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
ť
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
is_constant(*
T0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
ż
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
Î
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat*
T0*
is_constant(
ý
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
ď
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1MatMulegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
Ľ
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
dtype0*
_output_shapes	
:
Ń
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc*
is_constant( *
_output_shapes	
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
Ě
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:: 
ý
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*"
_output_shapes
::*
T0
´
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/AddAddYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch:1Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
ë
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:*
T0
ß
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:*
T0
Ş
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/RankConst^gradients/Sub*
_output_shapes
: *
dtype0*
value	B :

]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis
ś
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
ż
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPush	StackPush`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnterIencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis
É
igradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
T0*
is_constant(
î
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopStackPopigradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnter^gradients/Sub*
	elem_type0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
_output_shapes
: 
Ŕ
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/modFloorMod`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
ş
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
š
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Shape_1Shapeencoder_1/rnn/while/Identity_3*
T0*
out_type0*
_output_shapes
:
Ř
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*.
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
Ą
lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
Đ
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopStackPoplgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
Î
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeNShapeNcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop*
N*
T0* 
_output_shapes
::*
out_type0
Ž
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffsetConcatOffsetWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/modZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
´
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/SliceSliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffsetZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN* 
_output_shapes
:
*
Index0*
T0
Â
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Slice_1SliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMulbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffset:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Index0*
T0
¸
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
ě
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
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/AddAddbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:


ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIterationNextIteration]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
*
T0
ö
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3Exit`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
*
T0
É
\gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterencoder_1/rnn/TensorArray_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
T0
ô
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
Vgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3\gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*
source	gradients*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes

::
č
Rgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1W^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: 
ů
^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_accStack*
	elem_type0*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*

stack_name *
_output_shapes
:
­
agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnterRefEnter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
T0

bgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPush	StackPushagradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnterencoder_1/rnn/while/Identity^gradients/Add*
_output_shapes
:*
swap_memory( *Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
T0
Ŕ
jgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop/RefEnterRefEnter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
T0*
is_constant(
ĺ
agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopStackPopjgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop/RefEnter^gradients/Sub*
	elem_type0*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
: 
Š
Xgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Vgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/SliceRgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
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
Bgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
_output_shapes
: *
dtype0
¤
Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterBgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
is_constant( *
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 

Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Jgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
_output_shapes
: : *
N*
T0
Ë
Cgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*
T0*
_output_shapes
: : 

@gradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddEgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Xgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
ž
Jgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration@gradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
_output_shapes
: *
T0
˛
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
Ř
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
kgradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3ygradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3&encoder_1/rnn/TensorArrayUnstack/rangeugradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*%
_output_shapes
:*
element_shape:*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
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
Ő
-gradients/encoder/conv1d/Squeeze_grad/ReshapeReshape,gradients/encoder_1/transpose_grad/transpose+gradients/encoder/conv1d/Squeeze_grad/Shape*
T0*
Tshape0*)
_output_shapes
:

*gradients/encoder/conv1d/Conv2D_grad/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
Ň
8gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/encoder/conv1d/Conv2D_grad/Shapeencoder/conv1d/ExpandDims_1-gradients/encoder/conv1d/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*(
_output_shapes
:

,gradients/encoder/conv1d/Conv2D_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            
Ó
9gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/conv1d/ExpandDims,gradients/encoder/conv1d/Conv2D_grad/Shape_1-gradients/encoder/conv1d/Squeeze_grad/Reshape*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*'
_output_shapes
:

0gradients/encoder/conv1d/ExpandDims_1_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         
ć
2gradients/encoder/conv1d/ExpandDims_1_grad/ReshapeReshape9gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropFilter0gradients/encoder/conv1d/ExpandDims_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:
¸
global_norm/L2LossL2Loss2gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*
T0*
_output_shapes
: *E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape
ş
global_norm/L2Loss_1L2Loss2gradients/encoder_1/initial_state_0_tiled_grad/Sum*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
: *
T0
ş
global_norm/L2Loss_2L2Loss2gradients/encoder_1/initial_state_1_tiled_grad/Sum*
T0*
_output_shapes
: *E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum
ş
global_norm/L2Loss_3L2Loss2gradients/encoder_1/initial_state_2_tiled_grad/Sum*
T0*
_output_shapes
: *E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum
ş
global_norm/L2Loss_4L2Loss2gradients/encoder_1/initial_state_3_tiled_grad/Sum*
_output_shapes
: *E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
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
global_norm/L2Loss_7L2Lossagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*
_output_shapes
: *t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3

global_norm/L2Loss_8L2LossXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: *k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
Ę
global_norm/L2Loss_9L2Loss:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
_output_shapes
: *
T0
Ă
global_norm/L2Loss_10L2Loss6gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
T0*
_output_shapes
: *I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad
Ä
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3global_norm/L2Loss_4global_norm/L2Loss_5global_norm/L2Loss_6global_norm/L2Loss_7global_norm/L2Loss_8global_norm/L2Loss_9global_norm/L2Loss_10*
N*
T0*
_output_shapes
:*

axis 
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
 *  ?*
_output_shapes
: *
dtype0
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
 *  ?
d
clip_by_global_norm/truediv_1/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0

clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
_output_shapes
: *
T0

clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
T0*
_output_shapes
: 
^
clip_by_global_norm/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
_output_shapes
: *
T0
â
clip_by_global_norm/mul_1Mul2gradients/encoder/conv1d/ExpandDims_1_grad/Reshapeclip_by_global_norm/mul*
T0*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*#
_output_shapes
:
Ć
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*
T0*#
_output_shapes
:*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape
Ţ
clip_by_global_norm/mul_2Mul2gradients/encoder_1/initial_state_0_tiled_grad/Sumclip_by_global_norm/mul*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
:	*
T0
Â
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
:	*
T0
Ţ
clip_by_global_norm/mul_3Mul2gradients/encoder_1/initial_state_1_tiled_grad/Sumclip_by_global_norm/mul*
_output_shapes
:	*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
T0
Â
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3*
_output_shapes
:	*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
T0
Ţ
clip_by_global_norm/mul_4Mul2gradients/encoder_1/initial_state_2_tiled_grad/Sumclip_by_global_norm/mul*
T0*
_output_shapes
:	*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum
Â
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum*
_output_shapes
:	
Ţ
clip_by_global_norm/mul_5Mul2gradients/encoder_1/initial_state_3_tiled_grad/Sumclip_by_global_norm/mul*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
_output_shapes
:	*
T0
Â
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
_output_shapes
:	*
T0
˝
clip_by_global_norm/mul_6Mulagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:

ň
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*
T0* 
_output_shapes
:
*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3
Ś
clip_by_global_norm/mul_7MulXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:*
T0
ä
*clip_by_global_norm/clip_by_global_norm/_6Identityclip_by_global_norm/mul_7*
T0*
_output_shapes	
:*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3
˝
clip_by_global_norm/mul_8Mulagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0* 
_output_shapes
:
*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3
ň
*clip_by_global_norm/clip_by_global_norm/_7Identityclip_by_global_norm/mul_8*
T0*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:

Ś
clip_by_global_norm/mul_9MulXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:
ä
*clip_by_global_norm/clip_by_global_norm/_8Identityclip_by_global_norm/mul_9*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:*
T0
ď
clip_by_global_norm/mul_10Mul:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1clip_by_global_norm/mul*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
Ë
*clip_by_global_norm/clip_by_global_norm/_9Identityclip_by_global_norm/mul_10*
T0*
_output_shapes
:	*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1
â
clip_by_global_norm/mul_11Mul6gradients/output_projection/xw_plus_b_grad/BiasAddGradclip_by_global_norm/mul*I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
_output_shapes
:*
T0
Ă
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
grad_norms/grad_normsScalarSummarygrad_norms/grad_norms/tagsglobal_norm/global_norm*
_output_shapes
: *
T0
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
VariableV2*
shared_name *
_class
loc:@input_embed*
	container *
shape: *
dtype0*
_output_shapes
: 
Ž
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
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wž?*
_class
loc:@input_embed

beta2_power
VariableV2*
shared_name *
_class
loc:@input_embed*
	container *
shape: *
dtype0*
_output_shapes
: 
Ž
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
zerosConst*"
valueB*    *
dtype0*#
_output_shapes
:
Ž
input_embed/Adam
VariableV2*#
_output_shapes
:*
dtype0*
shape:*
	container *
_class
loc:@input_embed*
shared_name 
ą
input_embed/Adam/AssignAssigninput_embed/Adamzeros*
use_locking(*
validate_shape(*
T0*#
_output_shapes
:*
_class
loc:@input_embed

input_embed/Adam/readIdentityinput_embed/Adam*#
_output_shapes
:*
_class
loc:@input_embed*
T0
f
zeros_1Const*"
valueB*    *#
_output_shapes
:*
dtype0
°
input_embed/Adam_1
VariableV2*
_class
loc:@input_embed*#
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
ˇ
input_embed/Adam_1/AssignAssigninput_embed/Adam_1zeros_1*
use_locking(*
validate_shape(*
T0*#
_output_shapes
:*
_class
loc:@input_embed

input_embed/Adam_1/readIdentityinput_embed/Adam_1*
_class
loc:@input_embed*#
_output_shapes
:*
T0
^
zeros_2Const*
_output_shapes
:	*
dtype0*
valueB	*    
ž
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
#encoder/initial_state_0/Adam/AssignAssignencoder/initial_state_0/Adamzeros_2**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
Ą
!encoder/initial_state_0/Adam/readIdentityencoder/initial_state_0/Adam*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0*
T0
^
zeros_3Const*
valueB	*    *
_output_shapes
:	*
dtype0
Ŕ
encoder/initial_state_0/Adam_1
VariableV2*
shape:	*
_output_shapes
:	*
shared_name **
_class 
loc:@encoder/initial_state_0*
dtype0*
	container 
×
%encoder/initial_state_0/Adam_1/AssignAssignencoder/initial_state_0/Adam_1zeros_3*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(
Ľ
#encoder/initial_state_0/Adam_1/readIdentityencoder/initial_state_0/Adam_1*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0*
T0
^
zeros_4Const*
valueB	*    *
dtype0*
_output_shapes
:	
ž
encoder/initial_state_1/Adam
VariableV2*
shared_name *
shape:	*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1*
dtype0*
	container 
Ó
#encoder/initial_state_1/Adam/AssignAssignencoder/initial_state_1/Adamzeros_4*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_1*
T0*
use_locking(
Ą
!encoder/initial_state_1/Adam/readIdentityencoder/initial_state_1/Adam*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1*
T0
^
zeros_5Const*
_output_shapes
:	*
dtype0*
valueB	*    
Ŕ
encoder/initial_state_1/Adam_1
VariableV2*
shared_name *
shape:	*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1*
dtype0*
	container 
×
%encoder/initial_state_1/Adam_1/AssignAssignencoder/initial_state_1/Adam_1zeros_5*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	
Ľ
#encoder/initial_state_1/Adam_1/readIdentityencoder/initial_state_1/Adam_1**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	*
T0
^
zeros_6Const*
dtype0*
_output_shapes
:	*
valueB	*    
ž
encoder/initial_state_2/Adam
VariableV2*
shared_name *
shape:	*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2*
dtype0*
	container 
Ó
#encoder/initial_state_2/Adam/AssignAssignencoder/initial_state_2/Adamzeros_6*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_2*
T0*
use_locking(
Ą
!encoder/initial_state_2/Adam/readIdentityencoder/initial_state_2/Adam**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	*
T0
^
zeros_7Const*
valueB	*    *
_output_shapes
:	*
dtype0
Ŕ
encoder/initial_state_2/Adam_1
VariableV2*
_output_shapes
:	*
dtype0*
shape:	*
	container **
_class 
loc:@encoder/initial_state_2*
shared_name 
×
%encoder/initial_state_2/Adam_1/AssignAssignencoder/initial_state_2/Adam_1zeros_7**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
Ľ
#encoder/initial_state_2/Adam_1/readIdentityencoder/initial_state_2/Adam_1*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2
^
zeros_8Const*
valueB	*    *
dtype0*
_output_shapes
:	
ž
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
Ą
!encoder/initial_state_3/Adam/readIdentityencoder/initial_state_3/Adam*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3*
T0
^
zeros_9Const*
_output_shapes
:	*
dtype0*
valueB	*    
Ŕ
encoder/initial_state_3/Adam_1
VariableV2*
shared_name **
_class 
loc:@encoder/initial_state_3*
	container *
shape:	*
dtype0*
_output_shapes
:	
×
%encoder/initial_state_3/Adam_1/AssignAssignencoder/initial_state_3/Adam_1zeros_9*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_3*
T0*
use_locking(
Ľ
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
ř
8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam
VariableV2*
shared_name *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
	container *
shape:
*
dtype0* 
_output_shapes
:

Š
?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/AssignAssign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamzeros_10* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0*
use_locking(
ö
=encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/readIdentity8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0
a
zeros_11Const* 
_output_shapes
:
*
dtype0*
valueB
*    
ú
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1
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
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
­
Aencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/AssignAssign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1zeros_11* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0*
use_locking(
ú
?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/readIdentity:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
W
zeros_12Const*
_output_shapes	
:*
dtype0*
valueB*    
ě
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam
VariableV2*
_output_shapes	
:*
dtype0*
shape:*
	container *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
shared_name 
Ą
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adamzeros_12*
_output_shapes	
:*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
T0*
use_locking(
î
<encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/readIdentity7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0
W
zeros_13Const*
valueB*    *
dtype0*
_output_shapes	
:
î
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1
VariableV2*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
	container *
shape:*
dtype0*
_output_shapes	
:
Ľ
@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1zeros_13*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
ň
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
dtype0* 
_output_shapes
:

ř
8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam
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
Š
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
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
a
zeros_15Const*
valueB
*    * 
_output_shapes
:
*
dtype0
ú
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1
VariableV2*
shared_name *
shape:
* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
	container 
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
ú
?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/readIdentity:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:

W
zeros_16Const*
valueB*    *
_output_shapes	
:*
dtype0
ě
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam
VariableV2*
shape:*
_output_shapes	
:*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
dtype0*
	container 
Ą
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adamzeros_16*
_output_shapes	
:*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
use_locking(
î
<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/readIdentity7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:
W
zeros_17Const*
dtype0*
_output_shapes	
:*
valueB*    
î
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
Ľ
@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1zeros_17*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
ň
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/readIdentity9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
_
zeros_18Const*
valueB	*    *
dtype0*
_output_shapes
:	
ś
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
output_projection/W/Adam/readIdentityoutput_projection/W/Adam*
_output_shapes
:	*&
_class
loc:@output_projection/W*
T0
_
zeros_19Const*
valueB	*    *
_output_shapes
:	*
dtype0
¸
output_projection/W/Adam_1
VariableV2*&
_class
loc:@output_projection/W*
_output_shapes
:	*
shape:	*
dtype0*
shared_name *
	container 
Ě
!output_projection/W/Adam_1/AssignAssignoutput_projection/W/Adam_1zeros_19*
use_locking(*
T0*&
_class
loc:@output_projection/W*
validate_shape(*
_output_shapes
:	

output_projection/W/Adam_1/readIdentityoutput_projection/W/Adam_1*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0
U
zeros_20Const*
valueB*    *
dtype0*
_output_shapes
:
Ź
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
Ă
output_projection/b/Adam/AssignAssignoutput_projection/b/Adamzeros_20*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

output_projection/b/Adam/readIdentityoutput_projection/b/Adam*
T0*
_output_shapes
:*&
_class
loc:@output_projection/b
U
zeros_21Const*
dtype0*
_output_shapes
:*
valueB*    
Ž
output_projection/b/Adam_1
VariableV2*
shape:*
_output_shapes
:*
shared_name *&
_class
loc:@output_projection/b*
dtype0*
	container 
Ç
!output_projection/b/Adam_1/AssignAssignoutput_projection/b/Adam_1zeros_21*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(

output_projection/b/Adam_1/readIdentityoutput_projection/b/Adam_1*
T0*
_output_shapes
:*&
_class
loc:@output_projection/b
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wž?
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
Ç
!Adam/update_input_embed/ApplyAdam	ApplyAdaminput_embedinput_embed/Adaminput_embed/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
_class
loc:@input_embed*#
_output_shapes
:*
T0*
use_locking( 
˙
-Adam/update_encoder/initial_state_0/ApplyAdam	ApplyAdamencoder/initial_state_0encoder/initial_state_0/Adamencoder/initial_state_0/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_locking( *
T0**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	
˙
-Adam/update_encoder/initial_state_1/ApplyAdam	ApplyAdamencoder/initial_state_1encoder/initial_state_1/Adamencoder/initial_state_1/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
use_locking( *
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1
˙
-Adam/update_encoder/initial_state_2/ApplyAdam	ApplyAdamencoder/initial_state_2encoder/initial_state_2/Adamencoder/initial_state_2/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
use_locking( *
T0**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	
˙
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
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:


HAdam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdam	ApplyAdam2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_6*
use_locking( *
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases

IAdam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdam	ApplyAdam3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_7* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0*
use_locking( 

HAdam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam	ApplyAdam2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_8*
use_locking( *
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
ë
)Adam/update_output_projection/W/ApplyAdam	ApplyAdamoutput_projection/Woutput_projection/W/Adamoutput_projection/W/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_9*
use_locking( *
T0*
_output_shapes
:	*&
_class
loc:@output_projection/W
ç
)Adam/update_output_projection/b/ApplyAdam	ApplyAdamoutput_projection/boutput_projection/b/Adamoutput_projection/b/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_10*
use_locking( *
T0*&
_class
loc:@output_projection/b*
_output_shapes
:
Ř
Adam/mulMulbeta1_power/read
Adam/beta1"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam*
T0*
_class
loc:@input_embed*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
_class
loc:@input_embed*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
Ú

Adam/mul_1Mulbeta2_power/read
Adam/beta2"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam*
T0*
_class
loc:@input_embed*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@input_embed*
validate_shape(*
_output_shapes
: 

Adam/updateNoOp"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam^Adam/Assign^Adam/Assign_1
w

Adam/valueConst^Adam/update*
_output_shapes
: *
dtype0*
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
Z
train_loss/tagsConst*
_output_shapes
: *
dtype0*
valueB B
train_loss
X

train_lossScalarSummarytrain_loss/tags	loss/Mean*
_output_shapes
: *
T0
b
train_accuracy/tagsConst*
valueB Btrain_accuracy*
_output_shapes
: *
dtype0
h
train_accuracyScalarSummarytrain_accuracy/tagsaccuracy/accuracy*
_output_shapes
: *
T0
_
Merge/MergeSummaryMergeSummary
train_losstrain_accuracy*
N*
_output_shapes
: 
P

save/ConstConst*
_output_shapes
: *
dtype0*
valueB Bmodel
Ń

save/SaveV2/tensor_namesConst*
_output_shapes
:$*
dtype0*

valueú	B÷	$BVariableBbeta1_powerBbeta2_powerBencoder/initial_state_0Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1Binput_embedBinput_embed/AdamBinput_embed/Adam_1Boutput_projection/WBoutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/bBoutput_projection/b/AdamBoutput_projection/b/Adam_1
Ť
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
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2

save/AssignAssignVariablesave/RestoreV2*
_output_shapes
: *
validate_shape(*
_class
loc:@Variable*
T0*
use_locking(
q
save/RestoreV2_1/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
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
Á
save/Assign_3Assignencoder/initial_state_0save/RestoreV2_3*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(

save/RestoreV2_4/tensor_namesConst*
dtype0*
_output_shapes
:*1
value(B&Bencoder/initial_state_0/Adam
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
Ć
save/Assign_4Assignencoder/initial_state_0/Adamsave/RestoreV2_4*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(

save/RestoreV2_5/tensor_namesConst*
dtype0*
_output_shapes
:*3
value*B(Bencoder/initial_state_0/Adam_1
j
!save/RestoreV2_5/shape_and_slicesConst*
_output_shapes
:*
dtype0*
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
use_locking(*
T0**
_class 
loc:@encoder/initial_state_0*
validate_shape(*
_output_shapes
:	
}
save/RestoreV2_6/tensor_namesConst*,
value#B!Bencoder/initial_state_1*
dtype0*
_output_shapes
:
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
Á
save/Assign_6Assignencoder/initial_state_1save/RestoreV2_6**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

save/RestoreV2_7/tensor_namesConst*
dtype0*
_output_shapes
:*1
value(B&Bencoder/initial_state_1/Adam
j
!save/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ć
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
value*B(Bencoder/initial_state_1/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
Č
save/Assign_8Assignencoder/initial_state_1/Adam_1save/RestoreV2_8*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1
}
save/RestoreV2_9/tensor_namesConst*,
value#B!Bencoder/initial_state_2*
_output_shapes
:*
dtype0
j
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Á
save/Assign_9Assignencoder/initial_state_2save/RestoreV2_9*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_2*
T0*
use_locking(

save/RestoreV2_10/tensor_namesConst*
dtype0*
_output_shapes
:*1
value(B&Bencoder/initial_state_2/Adam
k
"save/RestoreV2_10/shape_and_slicesConst*
dtype0*
_output_shapes
:*
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
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
Ę
save/Assign_11Assignencoder/initial_state_2/Adam_1save/RestoreV2_11*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_2*
T0*
use_locking(
~
save/RestoreV2_12/tensor_namesConst*
_output_shapes
:*
dtype0*,
value#B!Bencoder/initial_state_3
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
Ă
save/Assign_12Assignencoder/initial_state_3save/RestoreV2_12*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	

save/RestoreV2_13/tensor_namesConst*
_output_shapes
:*
dtype0*1
value(B&Bencoder/initial_state_3/Adam
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
Č
save/Assign_13Assignencoder/initial_state_3/Adamsave/RestoreV2_13*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	

save/RestoreV2_14/tensor_namesConst*3
value*B(Bencoder/initial_state_3/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save/Assign_14Assignencoder/initial_state_3/Adam_1save/RestoreV2_14**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

save/RestoreV2_15/tensor_namesConst*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
dtype0*
_output_shapes
:
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
ő
save/Assign_15Assign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasessave/RestoreV2_15*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(

save/RestoreV2_16/tensor_namesConst*
dtype0*
_output_shapes
:*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
ú
save/Assign_16Assign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adamsave/RestoreV2_16*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
 
save/RestoreV2_17/tensor_namesConst*N
valueEBCB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*
dtype0*
_output_shapes
:
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
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases

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

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
_output_shapes
:*
dtypes
2
ü
save/Assign_18Assign3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightssave/RestoreV2_18*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(

save/RestoreV2_19/tensor_namesConst*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_19/shape_and_slicesConst*
_output_shapes
:*
dtype0*
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
save/Assign_19Assign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamsave/RestoreV2_19*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ą
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
save/Assign_20Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1save/RestoreV2_20*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(

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

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ő
save/Assign_21Assign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasessave/RestoreV2_21*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases

save/RestoreV2_22/tensor_namesConst*
_output_shapes
:*
dtype0*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam
k
"save/RestoreV2_22/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
_output_shapes
:*
dtypes
2
ú
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
valueEBCB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
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
save/RestoreV2_24/tensor_namesConst*
_output_shapes
:*
dtype0*H
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
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
validate_shape(* 
_output_shapes
:


save/RestoreV2_25/tensor_namesConst*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_25/shape_and_slicesConst*
_output_shapes
:*
dtype0*
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
save/Assign_25Assign8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adamsave/RestoreV2_25* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0*
use_locking(
Ą
save/RestoreV2_26/tensor_namesConst*
dtype0*
_output_shapes
:*O
valueFBDB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1
k
"save/RestoreV2_26/shape_and_slicesConst*
dtype0*
_output_shapes
:*
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
valueBBinput_embed*
dtype0*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_27Assigninput_embedsave/RestoreV2_27*#
_output_shapes
:*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
w
save/RestoreV2_28/tensor_namesConst*%
valueBBinput_embed/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
_output_shapes
:*
dtypes
2
´
save/Assign_28Assigninput_embed/Adamsave/RestoreV2_28*#
_output_shapes
:*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
y
save/RestoreV2_29/tensor_namesConst*
dtype0*
_output_shapes
:*'
valueBBinput_embed/Adam_1
k
"save/RestoreV2_29/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save/Assign_29Assigninput_embed/Adam_1save/RestoreV2_29*
_class
loc:@input_embed*#
_output_shapes
:*
T0*
validate_shape(*
use_locking(
z
save/RestoreV2_30/tensor_namesConst*(
valueBBoutput_projection/W*
dtype0*
_output_shapes
:
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
ť
save/Assign_30Assignoutput_projection/Wsave/RestoreV2_30*
_output_shapes
:	*
validate_shape(*&
_class
loc:@output_projection/W*
T0*
use_locking(

save/RestoreV2_31/tensor_namesConst*-
value$B"Boutput_projection/W/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_31/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
_output_shapes
:*
dtypes
2
Ŕ
save/Assign_31Assignoutput_projection/W/Adamsave/RestoreV2_31*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*&
_class
loc:@output_projection/W

save/RestoreV2_32/tensor_namesConst*/
value&B$Boutput_projection/W/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_32/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
_output_shapes
:*
dtypes
2
Â
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
"save/RestoreV2_33/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save/Assign_33Assignoutput_projection/bsave/RestoreV2_33*
use_locking(*
T0*&
_class
loc:@output_projection/b*
validate_shape(*
_output_shapes
:

save/RestoreV2_34/tensor_namesConst*-
value$B"Boutput_projection/b/Adam*
dtype0*
_output_shapes
:
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
ť
save/Assign_34Assignoutput_projection/b/Adamsave/RestoreV2_34*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(

save/RestoreV2_35/tensor_namesConst*
dtype0*
_output_shapes
:*/
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
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
_output_shapes
:*
dtypes
2
˝
save/Assign_35Assignoutput_projection/b/Adam_1save/RestoreV2_35*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(
đ
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35
f
train/total_loss/tagsConst*
_output_shapes
: *
dtype0*!
valueB Btrain/total_loss
c
train/total_lossScalarSummarytrain/total_loss/tagsloss/Sum*
_output_shapes
: *
T0
V
train/lr/tagsConst*
valueB Btrain/lr*
_output_shapes
: *
dtype0
X
train/lrScalarSummarytrain/lr/tagslearning_rate*
_output_shapes
: *
T0
a
Merge_1/MergeSummaryMergeSummarytrain/total_losstrain/lr*
N*
_output_shapes
: 
d
test/total_loss/tagsConst* 
valueB Btest/total_loss*
_output_shapes
: *
dtype0
a
test/total_lossScalarSummarytest/total_loss/tagsloss/Sum*
_output_shapes
: *
T0
V
Merge_2/MergeSummaryMergeSummarytest/total_loss*
N*
_output_shapes
: 
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Ó

save_1/SaveV2/tensor_namesConst*
_output_shapes
:$*
dtype0*

valueú	B÷	$BVariableBbeta1_powerBbeta2_powerBencoder/initial_state_0Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1Binput_embedBinput_embed/AdamBinput_embed/Adam_1Boutput_projection/WBoutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/bBoutput_projection/b/AdamBoutput_projection/b/Adam_1
­
save_1/SaveV2/shape_and_slicesConst*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:$*
dtype0

save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesVariablebeta1_powerbeta2_powerencoder/initial_state_0encoder/initial_state_0/Adamencoder/initial_state_0/Adam_1encoder/initial_state_1encoder/initial_state_1/Adamencoder/initial_state_1/Adam_1encoder/initial_state_2encoder/initial_state_2/Adamencoder/initial_state_2/Adam_1encoder/initial_state_3encoder/initial_state_3/Adamencoder/initial_state_3/Adam_12encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_13encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_12encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_13encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1input_embedinput_embed/Adaminput_embed/Adam_1output_projection/Woutput_projection/W/Adamoutput_projection/W/Adam_1output_projection/boutput_projection/b/Adamoutput_projection/b/Adam_1*2
dtypes(
&2$

save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
_output_shapes
: *
_class
loc:@save_1/Const*
T0
n
save_1/RestoreV2/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBBVariable
j
!save_1/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:

save_1/AssignAssignVariablesave_1/RestoreV2*
_output_shapes
: *
validate_shape(*
_class
loc:@Variable*
T0*
use_locking(
s
save_1/RestoreV2_1/tensor_namesConst*
_output_shapes
:*
dtype0* 
valueBBbeta1_power
l
#save_1/RestoreV2_1/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
¤
save_1/Assign_1Assignbeta1_powersave_1/RestoreV2_1*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@input_embed
s
save_1/RestoreV2_2/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save_1/RestoreV2_2	RestoreV2save_1/Constsave_1/RestoreV2_2/tensor_names#save_1/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
¤
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2_2*
_class
loc:@input_embed*
_output_shapes
: *
T0*
validate_shape(*
use_locking(

save_1/RestoreV2_3/tensor_namesConst*,
value#B!Bencoder/initial_state_0*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_3/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save_1/RestoreV2_3	RestoreV2save_1/Constsave_1/RestoreV2_3/tensor_names#save_1/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
Ĺ
save_1/Assign_3Assignencoder/initial_state_0save_1/RestoreV2_3*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_0*
validate_shape(*
_output_shapes
:	

save_1/RestoreV2_4/tensor_namesConst*
_output_shapes
:*
dtype0*1
value(B&Bencoder/initial_state_0/Adam
l
#save_1/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save_1/RestoreV2_4	RestoreV2save_1/Constsave_1/RestoreV2_4/tensor_names#save_1/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
Ę
save_1/Assign_4Assignencoder/initial_state_0/Adamsave_1/RestoreV2_4*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(

save_1/RestoreV2_5/tensor_namesConst*3
value*B(Bencoder/initial_state_0/Adam_1*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_5/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save_1/RestoreV2_5	RestoreV2save_1/Constsave_1/RestoreV2_5/tensor_names#save_1/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
Ě
save_1/Assign_5Assignencoder/initial_state_0/Adam_1save_1/RestoreV2_5*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0

save_1/RestoreV2_6/tensor_namesConst*
_output_shapes
:*
dtype0*,
value#B!Bencoder/initial_state_1
l
#save_1/RestoreV2_6/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save_1/RestoreV2_6	RestoreV2save_1/Constsave_1/RestoreV2_6/tensor_names#save_1/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
Ĺ
save_1/Assign_6Assignencoder/initial_state_1save_1/RestoreV2_6*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1

save_1/RestoreV2_7/tensor_namesConst*1
value(B&Bencoder/initial_state_1/Adam*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save_1/RestoreV2_7	RestoreV2save_1/Constsave_1/RestoreV2_7/tensor_names#save_1/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
Ę
save_1/Assign_7Assignencoder/initial_state_1/Adamsave_1/RestoreV2_7*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1

save_1/RestoreV2_8/tensor_namesConst*3
value*B(Bencoder/initial_state_1/Adam_1*
_output_shapes
:*
dtype0
l
#save_1/RestoreV2_8/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2_8	RestoreV2save_1/Constsave_1/RestoreV2_8/tensor_names#save_1/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ě
save_1/Assign_8Assignencoder/initial_state_1/Adam_1save_1/RestoreV2_8*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_1*
T0*
use_locking(

save_1/RestoreV2_9/tensor_namesConst*,
value#B!Bencoder/initial_state_2*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_9/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save_1/RestoreV2_9	RestoreV2save_1/Constsave_1/RestoreV2_9/tensor_names#save_1/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Ĺ
save_1/Assign_9Assignencoder/initial_state_2save_1/RestoreV2_9*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_2*
T0*
use_locking(

 save_1/RestoreV2_10/tensor_namesConst*1
value(B&Bencoder/initial_state_2/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_10/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_10	RestoreV2save_1/Const save_1/RestoreV2_10/tensor_names$save_1/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
Ě
save_1/Assign_10Assignencoder/initial_state_2/Adamsave_1/RestoreV2_10*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_2*
validate_shape(*
_output_shapes
:	

 save_1/RestoreV2_11/tensor_namesConst*3
value*B(Bencoder/initial_state_2/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_11/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_11	RestoreV2save_1/Const save_1/RestoreV2_11/tensor_names$save_1/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
Î
save_1/Assign_11Assignencoder/initial_state_2/Adam_1save_1/RestoreV2_11*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2

 save_1/RestoreV2_12/tensor_namesConst*,
value#B!Bencoder/initial_state_3*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_12	RestoreV2save_1/Const save_1/RestoreV2_12/tensor_names$save_1/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
Ç
save_1/Assign_12Assignencoder/initial_state_3save_1/RestoreV2_12*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3

 save_1/RestoreV2_13/tensor_namesConst*1
value(B&Bencoder/initial_state_3/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_13/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_13	RestoreV2save_1/Const save_1/RestoreV2_13/tensor_names$save_1/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Ě
save_1/Assign_13Assignencoder/initial_state_3/Adamsave_1/RestoreV2_13**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

 save_1/RestoreV2_14/tensor_namesConst*
_output_shapes
:*
dtype0*3
value*B(Bencoder/initial_state_3/Adam_1
m
$save_1/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_14	RestoreV2save_1/Const save_1/RestoreV2_14/tensor_names$save_1/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Î
save_1/Assign_14Assignencoder/initial_state_3/Adam_1save_1/RestoreV2_14*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	

 save_1/RestoreV2_15/tensor_namesConst*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes
:*
dtype0
m
$save_1/RestoreV2_15/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ą
save_1/RestoreV2_15	RestoreV2save_1/Const save_1/RestoreV2_15/tensor_names$save_1/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
ů
save_1/Assign_15Assign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasessave_1/RestoreV2_15*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
 
 save_1/RestoreV2_16/tensor_namesConst*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*
_output_shapes
:*
dtype0
m
$save_1/RestoreV2_16/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_16	RestoreV2save_1/Const save_1/RestoreV2_16/tensor_names$save_1/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
ţ
save_1/Assign_16Assign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adamsave_1/RestoreV2_16*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
˘
 save_1/RestoreV2_17/tensor_namesConst*
_output_shapes
:*
dtype0*N
valueEBCB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1
m
$save_1/RestoreV2_17/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ą
save_1/RestoreV2_17	RestoreV2save_1/Const save_1/RestoreV2_17/tensor_names$save_1/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:

save_1/Assign_17Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1save_1/RestoreV2_17*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(

 save_1/RestoreV2_18/tensor_namesConst*
_output_shapes
:*
dtype0*H
value?B=B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
m
$save_1/RestoreV2_18/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ą
save_1/RestoreV2_18	RestoreV2save_1/Const save_1/RestoreV2_18/tensor_names$save_1/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:

save_1/Assign_18Assign3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightssave_1/RestoreV2_18*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ą
 save_1/RestoreV2_19/tensor_namesConst*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*
_output_shapes
:*
dtype0
m
$save_1/RestoreV2_19/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_19	RestoreV2save_1/Const save_1/RestoreV2_19/tensor_names$save_1/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:

save_1/Assign_19Assign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamsave_1/RestoreV2_19*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ł
 save_1/RestoreV2_20/tensor_namesConst*
_output_shapes
:*
dtype0*O
valueFBDB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1
m
$save_1/RestoreV2_20/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_20	RestoreV2save_1/Const save_1/RestoreV2_20/tensor_names$save_1/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:

save_1/Assign_20Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1save_1/RestoreV2_20*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights

 save_1/RestoreV2_21/tensor_namesConst*
_output_shapes
:*
dtype0*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
m
$save_1/RestoreV2_21/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
Ą
save_1/RestoreV2_21	RestoreV2save_1/Const save_1/RestoreV2_21/tensor_names$save_1/RestoreV2_21/shape_and_slices*
_output_shapes
:*
dtypes
2
ů
save_1/Assign_21Assign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasessave_1/RestoreV2_21*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
 
 save_1/RestoreV2_22/tensor_namesConst*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_22/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_22	RestoreV2save_1/Const save_1/RestoreV2_22/tensor_names$save_1/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ţ
save_1/Assign_22Assign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adamsave_1/RestoreV2_22*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
validate_shape(*
_output_shapes	
:
˘
 save_1/RestoreV2_23/tensor_namesConst*N
valueEBCB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_23/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ą
save_1/RestoreV2_23	RestoreV2save_1/Const save_1/RestoreV2_23/tensor_names$save_1/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:

save_1/Assign_23Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1save_1/RestoreV2_23*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
validate_shape(*
_output_shapes	
:

 save_1/RestoreV2_24/tensor_namesConst*H
value?B=B3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_24	RestoreV2save_1/Const save_1/RestoreV2_24/tensor_names$save_1/RestoreV2_24/shape_and_slices*
_output_shapes
:*
dtypes
2

save_1/Assign_24Assign3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightssave_1/RestoreV2_24*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
validate_shape(* 
_output_shapes
:

Ą
 save_1/RestoreV2_25/tensor_namesConst*
dtype0*
_output_shapes
:*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam
m
$save_1/RestoreV2_25/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
Ą
save_1/RestoreV2_25	RestoreV2save_1/Const save_1/RestoreV2_25/tensor_names$save_1/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:

save_1/Assign_25Assign8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adamsave_1/RestoreV2_25*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
validate_shape(* 
_output_shapes
:

Ł
 save_1/RestoreV2_26/tensor_namesConst*O
valueFBDB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_26/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ą
save_1/RestoreV2_26	RestoreV2save_1/Const save_1/RestoreV2_26/tensor_names$save_1/RestoreV2_26/shape_and_slices*
_output_shapes
:*
dtypes
2

save_1/Assign_26Assign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1save_1/RestoreV2_26*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
t
 save_1/RestoreV2_27/tensor_namesConst*
dtype0*
_output_shapes
:* 
valueBBinput_embed
m
$save_1/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_27	RestoreV2save_1/Const save_1/RestoreV2_27/tensor_names$save_1/RestoreV2_27/shape_and_slices*
_output_shapes
:*
dtypes
2
ł
save_1/Assign_27Assigninput_embedsave_1/RestoreV2_27*
use_locking(*
T0*
_class
loc:@input_embed*
validate_shape(*#
_output_shapes
:
y
 save_1/RestoreV2_28/tensor_namesConst*
_output_shapes
:*
dtype0*%
valueBBinput_embed/Adam
m
$save_1/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_28	RestoreV2save_1/Const save_1/RestoreV2_28/tensor_names$save_1/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
save_1/Assign_28Assigninput_embed/Adamsave_1/RestoreV2_28*
use_locking(*
validate_shape(*
T0*#
_output_shapes
:*
_class
loc:@input_embed
{
 save_1/RestoreV2_29/tensor_namesConst*
dtype0*
_output_shapes
:*'
valueBBinput_embed/Adam_1
m
$save_1/RestoreV2_29/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ą
save_1/RestoreV2_29	RestoreV2save_1/Const save_1/RestoreV2_29/tensor_names$save_1/RestoreV2_29/shape_and_slices*
_output_shapes
:*
dtypes
2
ş
save_1/Assign_29Assigninput_embed/Adam_1save_1/RestoreV2_29*
use_locking(*
T0*
_class
loc:@input_embed*
validate_shape(*#
_output_shapes
:
|
 save_1/RestoreV2_30/tensor_namesConst*
dtype0*
_output_shapes
:*(
valueBBoutput_projection/W
m
$save_1/RestoreV2_30/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_30	RestoreV2save_1/Const save_1/RestoreV2_30/tensor_names$save_1/RestoreV2_30/shape_and_slices*
_output_shapes
:*
dtypes
2
ż
save_1/Assign_30Assignoutput_projection/Wsave_1/RestoreV2_30*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

 save_1/RestoreV2_31/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Boutput_projection/W/Adam
m
$save_1/RestoreV2_31/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_31	RestoreV2save_1/Const save_1/RestoreV2_31/tensor_names$save_1/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
Ä
save_1/Assign_31Assignoutput_projection/W/Adamsave_1/RestoreV2_31*
use_locking(*
T0*&
_class
loc:@output_projection/W*
validate_shape(*
_output_shapes
:	

 save_1/RestoreV2_32/tensor_namesConst*
_output_shapes
:*
dtype0*/
value&B$Boutput_projection/W/Adam_1
m
$save_1/RestoreV2_32/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_32	RestoreV2save_1/Const save_1/RestoreV2_32/tensor_names$save_1/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
Ć
save_1/Assign_32Assignoutput_projection/W/Adam_1save_1/RestoreV2_32*
use_locking(*
T0*&
_class
loc:@output_projection/W*
validate_shape(*
_output_shapes
:	
|
 save_1/RestoreV2_33/tensor_namesConst*(
valueBBoutput_projection/b*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_33	RestoreV2save_1/Const save_1/RestoreV2_33/tensor_names$save_1/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save_1/Assign_33Assignoutput_projection/bsave_1/RestoreV2_33*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*&
_class
loc:@output_projection/b

 save_1/RestoreV2_34/tensor_namesConst*-
value$B"Boutput_projection/b/Adam*
_output_shapes
:*
dtype0
m
$save_1/RestoreV2_34/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_34	RestoreV2save_1/Const save_1/RestoreV2_34/tensor_names$save_1/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
ż
save_1/Assign_34Assignoutput_projection/b/Adamsave_1/RestoreV2_34*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(

 save_1/RestoreV2_35/tensor_namesConst*/
value&B$Boutput_projection/b/Adam_1*
_output_shapes
:*
dtype0
m
$save_1/RestoreV2_35/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
Ą
save_1/RestoreV2_35	RestoreV2save_1/Const save_1/RestoreV2_35/tensor_names$save_1/RestoreV2_35/shape_and_slices*
_output_shapes
:*
dtypes
2
Á
save_1/Assign_35Assignoutput_projection/b/Adam_1save_1/RestoreV2_35*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*&
_class
loc:@output_projection/b
ş
save_1/restore_allNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35

4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedVariable*
_class
loc:@Variable*
_output_shapes
: *
dtype0
Ą
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedinput_embed*
_class
loc:@input_embed*
dtype0*
_output_shapes
: 
š
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedencoder/initial_state_0*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_0
š
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedencoder/initial_state_1**
_class 
loc:@encoder/initial_state_1*
dtype0*
_output_shapes
: 
š
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedencoder/initial_state_2**
_class 
loc:@encoder/initial_state_2*
_output_shapes
: *
dtype0
š
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedencoder/initial_state_3*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_3
ń
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
_output_shapes
: *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
ď
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
ń
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
_output_shapes
: *
dtype0
ď
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitialized2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
dtype0*
_output_shapes
: 
˛
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedoutput_projection/W*
dtype0*
_output_shapes
: *&
_class
loc:@output_projection/W
˛
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedoutput_projection/b*
dtype0*
_output_shapes
: *&
_class
loc:@output_projection/b
˘
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedbeta1_power*
_class
loc:@input_embed*
dtype0*
_output_shapes
: 
˘
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedbeta2_power*
_class
loc:@input_embed*
_output_shapes
: *
dtype0
§
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedinput_embed/Adam*
_class
loc:@input_embed*
_output_shapes
: *
dtype0
Š
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedinput_embed/Adam_1*
_class
loc:@input_embed*
_output_shapes
: *
dtype0
ż
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedencoder/initial_state_0/Adam*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_0
Á
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitializedencoder/initial_state_0/Adam_1**
_class 
loc:@encoder/initial_state_0*
_output_shapes
: *
dtype0
ż
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitializedencoder/initial_state_1/Adam*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_1
Á
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitializedencoder/initial_state_1/Adam_1*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_1
ż
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitializedencoder/initial_state_2/Adam*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_2
Á
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitializedencoder/initial_state_2/Adam_1**
_class 
loc:@encoder/initial_state_2*
dtype0*
_output_shapes
: 
ż
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitializedencoder/initial_state_3/Adam**
_class 
loc:@encoder/initial_state_3*
dtype0*
_output_shapes
: 
Á
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitializedencoder/initial_state_3/Adam_1*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_3
÷
7report_uninitialized_variables/IsVariableInitialized_24IsVariableInitialized8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
dtype0*
_output_shapes
: 
ů
7report_uninitialized_variables/IsVariableInitialized_25IsVariableInitialized:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
_output_shapes
: *
dtype0
ő
7report_uninitialized_variables/IsVariableInitialized_26IsVariableInitialized7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
dtype0*
_output_shapes
: 
÷
7report_uninitialized_variables/IsVariableInitialized_27IsVariableInitialized9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
÷
7report_uninitialized_variables/IsVariableInitialized_28IsVariableInitialized8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
_output_shapes
: 
ů
7report_uninitialized_variables/IsVariableInitialized_29IsVariableInitialized:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
ő
7report_uninitialized_variables/IsVariableInitialized_30IsVariableInitialized7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
÷
7report_uninitialized_variables/IsVariableInitialized_31IsVariableInitialized9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*
_output_shapes
: *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
ˇ
7report_uninitialized_variables/IsVariableInitialized_32IsVariableInitializedoutput_projection/W/Adam*&
_class
loc:@output_projection/W*
_output_shapes
: *
dtype0
š
7report_uninitialized_variables/IsVariableInitialized_33IsVariableInitializedoutput_projection/W/Adam_1*&
_class
loc:@output_projection/W*
_output_shapes
: *
dtype0
ˇ
7report_uninitialized_variables/IsVariableInitialized_34IsVariableInitializedoutput_projection/b/Adam*
_output_shapes
: *
dtype0*&
_class
loc:@output_projection/b
š
7report_uninitialized_variables/IsVariableInitialized_35IsVariableInitializedoutput_projection/b/Adam_1*
dtype0*
_output_shapes
: *&
_class
loc:@output_projection/b
Ţ
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
Ý

$report_uninitialized_variables/ConstConst*

valueú	B÷	$BVariableBinput_embedBencoder/initial_state_0Bencoder/initial_state_1Bencoder/initial_state_2Bencoder/initial_state_3B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesBoutput_projection/WBoutput_projection/bBbeta1_powerBbeta2_powerBinput_embed/AdamBinput_embed/Adam_1Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1B7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1Boutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/b/AdamBoutput_projection/b/Adam_1*
dtype0*
_output_shapes
:$
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:$*
dtype0*
_output_shapes
:

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ů
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
_output_shapes
:*
end_mask *
new_axis_mask *
ellipsis_mask *

begin_mask*
shrink_axis_mask *
T0*
Index0

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
ő
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:$

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
_output_shapes
:*
dtype0

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
á
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
Ż
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
N*
T0*
_output_shapes
:*

axis 
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ť
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
Ë
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
_output_shapes
:$*
T0

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
Ű
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:$*
T0


1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	*
squeeze_dims


2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

initNoOp^Variable/Assign^input_embed/Assign^encoder/initial_state_0/Assign^encoder/initial_state_1/Assign^encoder/initial_state_2/Assign^encoder/initial_state_3/Assign;^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Assign:^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Assign;^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Assign:^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Assign^output_projection/W/Assign^output_projection/b/Assign^beta1_power/Assign^beta2_power/Assign^input_embed/Adam/Assign^input_embed/Adam_1/Assign$^encoder/initial_state_0/Adam/Assign&^encoder/initial_state_0/Adam_1/Assign$^encoder/initial_state_1/Adam/Assign&^encoder/initial_state_1/Adam_1/Assign$^encoder/initial_state_2/Adam/Assign&^encoder/initial_state_2/Adam_1/Assign$^encoder/initial_state_3/Adam/Assign&^encoder/initial_state_3/Adam_1/Assign@^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/AssignB^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/Assign?^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/AssignA^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/Assign@^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/AssignB^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/Assign?^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/AssignA^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/Assign ^output_projection/W/Adam/Assign"^output_projection/W/Adam_1/Assign ^output_projection/b/Adam/Assign"^output_projection/b/Adam_1/Assign

init_1NoOp

init_all_tablesNoOp
-

group_depsNoOp^init_1^init_all_tables"Ä#J     Îaö	%ZWČľ?ÖAJ˝ś
DôC
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
ď
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
î
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
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
¸
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
Ttype*1.0.12v1.0.0-65-g4763edf-dirty¤
G
ConstConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
^
is_trainingPlaceholderWithDefaultConst*
_output_shapes
: *
dtype0
*
shape: 
b
inputPlaceholder*
shape:*
dtype0*$
_output_shapes
:
R
seq_lenPlaceholder*
_output_shapes	
:*
shape:*
dtype0
R
dec_lenPlaceholder*
shape:*
dtype0*
_output_shapes	
:
^
dec_targetsPlaceholder*
_output_shapes
:	*
dtype0*
shape:	
X
Variable/initial_valueConst*
value	B : *
_output_shapes
: *
dtype0
l
Variable
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
˘
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
Ą
,input_embed/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
_class
loc:@input_embed*!
valueB"         

*input_embed/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
_class
loc:@input_embed*
valueB
 *
×Ł˝

*input_embed/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
_class
loc:@input_embed*
valueB
 *
×Ł=
ç
4input_embed/Initializer/random_uniform/RandomUniformRandomUniform,input_embed/Initializer/random_uniform/shape*
T0*#
_output_shapes
:*

seed *
_class
loc:@input_embed*
dtype0*
seed2 
Ę
*input_embed/Initializer/random_uniform/subSub*input_embed/Initializer/random_uniform/max*input_embed/Initializer/random_uniform/min*
_output_shapes
: *
_class
loc:@input_embed*
T0
á
*input_embed/Initializer/random_uniform/mulMul4input_embed/Initializer/random_uniform/RandomUniform*input_embed/Initializer/random_uniform/sub*
T0*#
_output_shapes
:*
_class
loc:@input_embed
Ó
&input_embed/Initializer/random_uniformAdd*input_embed/Initializer/random_uniform/mul*input_embed/Initializer/random_uniform/min*
T0*#
_output_shapes
:*
_class
loc:@input_embed
Š
input_embed
VariableV2*
_class
loc:@input_embed*#
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Č
input_embed/AssignAssigninput_embed&input_embed/Initializer/random_uniform*#
_output_shapes
:*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
w
input_embed/readIdentityinput_embed*
T0*#
_output_shapes
:*
_class
loc:@input_embed
_
encoder/conv1d/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :

encoder/conv1d/ExpandDims
ExpandDimsinputencoder/conv1d/ExpandDims/dim*

Tdim0*(
_output_shapes
:*
T0
a
encoder/conv1d/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 

encoder/conv1d/ExpandDims_1
ExpandDimsinput_embed/readencoder/conv1d/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:
ă
encoder/conv1d/Conv2DConv2Dencoder/conv1d/ExpandDimsencoder/conv1d/ExpandDims_1*)
_output_shapes
:*
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
:*
T0
Z
ShapeConst*!
valueB"         *
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
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ů
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ź
)encoder/initial_state_0/Initializer/ConstConst**
_class 
loc:@encoder/initial_state_0*
valueB	*    *
_output_shapes
:	*
dtype0
š
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
ë
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
+encoder_1/initial_state_0_tiled/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
§
)encoder_1/initial_state_0_tiled/multiplesPackstrided_slice+encoder_1/initial_state_0_tiled/multiples/1*

axis *
_output_shapes
:*
T0*
N
ľ
encoder_1/initial_state_0_tiledTileencoder/initial_state_0/read)encoder_1/initial_state_0_tiled/multiples*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tmultiples0
Ź
)encoder/initial_state_1/Initializer/ConstConst**
_class 
loc:@encoder/initial_state_1*
valueB	*    *
dtype0*
_output_shapes
:	
š
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
ë
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
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1
m
+encoder_1/initial_state_1_tiled/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
§
)encoder_1/initial_state_1_tiled/multiplesPackstrided_slice+encoder_1/initial_state_1_tiled/multiples/1*

axis *
_output_shapes
:*
T0*
N
ľ
encoder_1/initial_state_1_tiledTileencoder/initial_state_1/read)encoder_1/initial_state_1_tiled/multiples*

Tmultiples0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
)encoder/initial_state_2/Initializer/ConstConst**
_class 
loc:@encoder/initial_state_2*
valueB	*    *
_output_shapes
:	*
dtype0
š
encoder/initial_state_2
VariableV2*
shared_name **
_class 
loc:@encoder/initial_state_2*
	container *
shape:	*
dtype0*
_output_shapes
:	
ë
encoder/initial_state_2/AssignAssignencoder/initial_state_2)encoder/initial_state_2/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2

encoder/initial_state_2/readIdentityencoder/initial_state_2**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	*
T0
m
+encoder_1/initial_state_2_tiled/multiples/1Const*
dtype0*
_output_shapes
: *
value	B :
§
)encoder_1/initial_state_2_tiled/multiplesPackstrided_slice+encoder_1/initial_state_2_tiled/multiples/1*
_output_shapes
:*
N*

axis *
T0
ľ
encoder_1/initial_state_2_tiledTileencoder/initial_state_2/read)encoder_1/initial_state_2_tiled/multiples*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tmultiples0
Ź
)encoder/initial_state_3/Initializer/ConstConst**
_class 
loc:@encoder/initial_state_3*
valueB	*    *
dtype0*
_output_shapes
:	
š
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
ë
encoder/initial_state_3/AssignAssignencoder/initial_state_3)encoder/initial_state_3/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3

encoder/initial_state_3/readIdentityencoder/initial_state_3*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3
m
+encoder_1/initial_state_3_tiled/multiples/1Const*
value	B :*
_output_shapes
: *
dtype0
§
)encoder_1/initial_state_3_tiled/multiplesPackstrided_slice+encoder_1/initial_state_3_tiled/multiples/1*
N*
T0*
_output_shapes
:*

axis 
ľ
encoder_1/initial_state_3_tiledTileencoder/initial_state_3/read)encoder_1/initial_state_3_tiled/multiples*

Tmultiples0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
encoder_1/transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          

encoder_1/transpose	Transposeencoder/conv1d/Squeezeencoder_1/transpose/perm*
Tperm0*
T0*%
_output_shapes
:
T
encoder_1/sequence_lengthIdentityseq_len*
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
!encoder_1/rnn/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
m
#encoder_1/rnn/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
m
#encoder_1/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ż
encoder_1/rnn/strided_sliceStridedSliceencoder_1/rnn/Shape!encoder_1/rnn/strided_slice/stack#encoder_1/rnn/strided_slice/stack_1#encoder_1/rnn/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask*
Index0*
T0*
end_mask *
_output_shapes
: *

begin_mask *
ellipsis_mask 
m
#encoder_1/rnn/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%encoder_1/rnn/strided_slice_1/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
o
%encoder_1/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ç
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
encoder_1/rnn/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
r
encoder_1/rnn/stackPackencoder_1/rnn/strided_slice*
N*
T0*
_output_shapes
:*

axis 
m
encoder_1/rnn/EqualEqualencoder_1/rnn/Shape_1encoder_1/rnn/stack*
T0*
_output_shapes
:
]
encoder_1/rnn/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
w
encoder_1/rnn/AllAllencoder_1/rnn/Equalencoder_1/rnn/Const*

Tidx0*
	keep_dims( *
_output_shapes
: 

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

"encoder_1/rnn/Assert/Assert/data_0Const*J
valueAB? B9Expected shape for Tensor encoder_1/sequence_length:0 is *
_output_shapes
: *
dtype0
s
"encoder_1/rnn/Assert/Assert/data_2Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
Ě
encoder_1/rnn/Assert/AssertAssertencoder_1/rnn/All"encoder_1/rnn/Assert/Assert/data_0encoder_1/rnn/stack"encoder_1/rnn/Assert/Assert/data_2encoder_1/rnn/Shape_1*
T
2*
	summarize

encoder_1/rnn/CheckSeqLenIdentityencoder_1/sequence_length^encoder_1/rnn/Assert/Assert*
_output_shapes	
:*
T0
j
encoder_1/rnn/Shape_2Const*
_output_shapes
:*
dtype0*!
valueB"         
m
#encoder_1/rnn/strided_slice_2/stackConst*
valueB: *
_output_shapes
:*
dtype0
o
%encoder_1/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%encoder_1/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
É
encoder_1/rnn/strided_slice_2StridedSliceencoder_1/rnn/Shape_2#encoder_1/rnn/strided_slice_2/stack%encoder_1/rnn/strided_slice_2/stack_1%encoder_1/rnn/strided_slice_2/stack_2*
_output_shapes
: *
end_mask *
new_axis_mask *
ellipsis_mask *

begin_mask *
shrink_axis_mask*
T0*
Index0
m
#encoder_1/rnn/strided_slice_3/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%encoder_1/rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
o
%encoder_1/rnn/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
É
encoder_1/rnn/strided_slice_3StridedSliceencoder_1/rnn/Shape_2#encoder_1/rnn/strided_slice_3/stack%encoder_1/rnn/strided_slice_3/stack_1%encoder_1/rnn/strided_slice_3/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0*
end_mask *
new_axis_mask *

begin_mask *
ellipsis_mask 
Z
encoder_1/rnn/stack_1/1Const*
value
B :*
_output_shapes
: *
dtype0

encoder_1/rnn/stack_1Packencoder_1/rnn/strided_slice_3encoder_1/rnn/stack_1/1*
T0*

axis *
N*
_output_shapes
:
^
encoder_1/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

encoder_1/rnn/zerosFillencoder_1/rnn/stack_1encoder_1/rnn/zeros/Const*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
encoder_1/rnn/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
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
encoder_1/rnn/MaxMaxencoder_1/rnn/CheckSeqLenencoder_1/rnn/Const_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
T
encoder_1/rnn/timeConst*
dtype0*
_output_shapes
: *
value	B : 
ô
encoder_1/rnn/TensorArrayTensorArrayV3encoder_1/rnn/strided_slice_2*
dynamic_size( *
clear_after_read(*
_output_shapes

::*
element_shape:*
dtype0*9
tensor_array_name$"encoder_1/rnn/dynamic_rnn/output_0
ő
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
valueB"         *
_output_shapes
:*
dtype0
~
4encoder_1/rnn/TensorArrayUnstack/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

.encoder_1/rnn/TensorArrayUnstack/strided_sliceStridedSlice&encoder_1/rnn/TensorArrayUnstack/Shape4encoder_1/rnn/TensorArrayUnstack/strided_slice/stack6encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_16encoder_1/rnn/TensorArrayUnstack/strided_slice/stack_2*
_output_shapes
: *
end_mask *
new_axis_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask*
Index0*
T0
n
,encoder_1/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
n
,encoder_1/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ě
&encoder_1/rnn/TensorArrayUnstack/rangeRange,encoder_1/rnn/TensorArrayUnstack/range/start.encoder_1/rnn/TensorArrayUnstack/strided_slice,encoder_1/rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
Hencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3encoder_1/rnn/TensorArray_1&encoder_1/rnn/TensorArrayUnstack/rangeencoder_1/transposeencoder_1/rnn/TensorArray_1:1*
_output_shapes
: *.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
T0
ż
encoder_1/rnn/while/EnterEnterencoder_1/rnn/time*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
Ě
encoder_1/rnn/while/Enter_1Enterencoder_1/rnn/TensorArray:1*
is_constant( *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
ŕ
encoder_1/rnn/while/Enter_2Enterencoder_1/initial_state_0_tiled*
parallel_iterations *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
ŕ
encoder_1/rnn/while/Enter_3Enterencoder_1/initial_state_1_tiled*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
ŕ
encoder_1/rnn/while/Enter_4Enterencoder_1/initial_state_2_tiled*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
ŕ
encoder_1/rnn/while/Enter_5Enterencoder_1/initial_state_3_tiled*
parallel_iterations *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
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
T0*
N
¤
encoder_1/rnn/while/Merge_2Mergeencoder_1/rnn/while/Enter_2#encoder_1/rnn/while/NextIteration_2**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
N*
T0
¤
encoder_1/rnn/while/Merge_3Mergeencoder_1/rnn/while/Enter_3#encoder_1/rnn/while/NextIteration_3**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N
¤
encoder_1/rnn/while/Merge_4Mergeencoder_1/rnn/while/Enter_4#encoder_1/rnn/while/NextIteration_4**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
N*
T0
¤
encoder_1/rnn/while/Merge_5Mergeencoder_1/rnn/while/Enter_5#encoder_1/rnn/while/NextIteration_5*
N*
T0**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
Ď
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
Ž
encoder_1/rnn/while/SwitchSwitchencoder_1/rnn/while/Mergeencoder_1/rnn/while/LoopCond*,
_class"
 loc:@encoder_1/rnn/while/Merge*
_output_shapes
: : *
T0
¸
encoder_1/rnn/while/Switch_1Switchencoder_1/rnn/while/Merge_1encoder_1/rnn/while/LoopCond*
_output_shapes

::*.
_class$
" loc:@encoder_1/rnn/while/Merge_1*
T0
Ř
encoder_1/rnn/while/Switch_2Switchencoder_1/rnn/while/Merge_2encoder_1/rnn/while/LoopCond*
T0*.
_class$
" loc:@encoder_1/rnn/while/Merge_2*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ř
encoder_1/rnn/while/Switch_3Switchencoder_1/rnn/while/Merge_3encoder_1/rnn/while/LoopCond*.
_class$
" loc:@encoder_1/rnn/while/Merge_3*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ř
encoder_1/rnn/while/Switch_4Switchencoder_1/rnn/while/Merge_4encoder_1/rnn/while/LoopCond*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@encoder_1/rnn/while/Merge_4
Ř
encoder_1/rnn/while/Switch_5Switchencoder_1/rnn/while/Merge_5encoder_1/rnn/while/LoopCond*.
_class$
" loc:@encoder_1/rnn/while/Merge_5*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
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
encoder_1/rnn/while/Identity_2Identityencoder_1/rnn/while/Switch_2:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
}
encoder_1/rnn/while/Identity_3Identityencoder_1/rnn/while/Switch_3:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
}
encoder_1/rnn/while/Identity_4Identityencoder_1/rnn/while/Switch_4:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
encoder_1/rnn/while/Identity_5Identityencoder_1/rnn/while/Switch_5:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

+encoder_1/rnn/while/TensorArrayReadV3/EnterEnterencoder_1/rnn/TensorArray_1*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
š
-encoder_1/rnn/while/TensorArrayReadV3/Enter_1EnterHencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
parallel_iterations *
is_constant(*
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/

%encoder_1/rnn/while/TensorArrayReadV3TensorArrayReadV3+encoder_1/rnn/while/TensorArrayReadV3/Enterencoder_1/rnn/while/Identity-encoder_1/rnn/while/TensorArrayReadV3/Enter_1*
dtype0* 
_output_shapes
:
*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
í
Tencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB"      *
dtype0*
_output_shapes
:
ß
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/minConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
valueB
 *
×Ł˝*
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
×Ł=
Ü
\encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/RandomUniformRandomUniformTencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/shape*
seed2 *
T0*

seed *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:

ę
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/subSubRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/maxRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
_output_shapes
: 
ţ
Rencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/mulMul\encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/RandomUniformRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:

đ
Nencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniformAddRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/mulRencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
ó
3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
VariableV2*
shared_name *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
	container *
shape:
*
dtype0* 
_output_shapes
:

ĺ
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AssignAssign3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsNencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Initializer/random_uniform*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
¤
8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/readIdentity3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0* 
_output_shapes
:

Ş
Iencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axisConst^encoder_1/rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :
˘
Dencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concatConcatV2%encoder_1/rnn/while/TensorArrayReadV3encoder_1/rnn/while/Identity_3Iencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*

Tidx0*
T0*
N* 
_output_shapes
:

 
Jencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read*
parallel_iterations *
T0* 
_output_shapes
:
*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(
ą
Dencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMulMatMulDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concatJencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter*
transpose_b( * 
_output_shapes
:
*
transpose_a( *
T0
Ú
Dencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Initializer/ConstConst*
_output_shapes	
:*
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
valueB*    
ç
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
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AssignAssign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesDencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases

7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/readIdentity2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
T0*
_output_shapes	
:

Aencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/EnterEnter7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read*
is_constant(*
_output_shapes	
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 

;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAddBiasAddDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMulAencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC* 
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
 *  ?*
_output_shapes
: *
dtype0
á
7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/addAdd;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add/y* 
_output_shapes
:
*
T0
Ş
;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/SigmoidSigmoid7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add* 
_output_shapes
:
*
T0
Ć
7encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mulMul;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoidencoder_1/rnn/while/Identity_2*
T0* 
_output_shapes
:

Ž
=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1Sigmoid9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split*
T0* 
_output_shapes
:

¨
8encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/TanhTanh;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split:1*
T0* 
_output_shapes
:

ä
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1Mul=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_18encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh* 
_output_shapes
:
*
T0
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
¨
:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1Tanh9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1* 
_output_shapes
:
*
T0
ć
9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2Mul=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0* 
_output_shapes
:

í
Tencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB"      *
_output_shapes
:*
dtype0
ß
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB
 *
×Ł˝
ß
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
valueB
 *
×Ł=
Ü
\encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/RandomUniformRandomUniformTencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/shape* 
_output_shapes
:
*
dtype0*
seed2 *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0*

seed 
ę
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/subSubRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/maxRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
ţ
Rencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/mulMul\encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/RandomUniformRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:

đ
Nencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniformAddRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/mulRencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Initializer/random_uniform/min*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
T0
ó
3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
VariableV2*
shared_name *
shape:
* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
	container 
ĺ
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
Ş
Iencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axisConst^encoder_1/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
ś
Dencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concatConcatV29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2encoder_1/rnn/while/Identity_5Iencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis* 
_output_shapes
:
*
T0*

Tidx0*
N
 
Jencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/EnterEnter8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read*
is_constant(* 
_output_shapes
:
*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
ą
Dencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMulMatMulDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concatJencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
Ú
Dencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Initializer/ConstConst*
dtype0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
valueB*    
ç
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
Cencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dimConst^encoder_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
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
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add/yConst^encoder_1/rnn/while/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?
á
7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/addAdd;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:29encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add/y* 
_output_shapes
:
*
T0
Ş
;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/SigmoidSigmoid7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add*
T0* 
_output_shapes
:

Ć
7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mulMul;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoidencoder_1/rnn/while/Identity_4* 
_output_shapes
:
*
T0
Ž
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1Sigmoid9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split* 
_output_shapes
:
*
T0
¨
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
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1Add7encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1*
T0* 
_output_shapes
:

°
=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2Sigmoid;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split:3*
T0* 
_output_shapes
:

¨
:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1Tanh9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1* 
_output_shapes
:
*
T0
ć
9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2Mul=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1* 
_output_shapes
:
*
T0
Ř
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
Ů
 encoder_1/rnn/while/Select/EnterEnterencoder_1/rnn/zeros*
parallel_iterations *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(
Î
encoder_1/rnn/while/SelectSelect encoder_1/rnn/while/GreaterEqual encoder_1/rnn/while/Select/Enter9encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0* 
_output_shapes
:


"encoder_1/rnn/while/GreaterEqual_1GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:
Đ
encoder_1/rnn/while/Select_1Select"encoder_1/rnn/while/GreaterEqual_1encoder_1/rnn/while/Identity_29encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1* 
_output_shapes
:
*
T0

"encoder_1/rnn/while/GreaterEqual_2GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:
Đ
encoder_1/rnn/while/Select_2Select"encoder_1/rnn/while/GreaterEqual_2encoder_1/rnn/while/Identity_39encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
T0* 
_output_shapes
:


"encoder_1/rnn/while/GreaterEqual_3GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes	
:
Đ
encoder_1/rnn/while/Select_3Select"encoder_1/rnn/while/GreaterEqual_3encoder_1/rnn/while/Identity_49encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1*
T0* 
_output_shapes
:


"encoder_1/rnn/while/GreaterEqual_4GreaterEqualencoder_1/rnn/while/Identity&encoder_1/rnn/while/GreaterEqual/Enter*
_output_shapes	
:*
T0
Đ
encoder_1/rnn/while/Select_4Select"encoder_1/rnn/while/GreaterEqual_4encoder_1/rnn/while/Identity_59encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2* 
_output_shapes
:
*
T0

=encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterencoder_1/rnn/TensorArray*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*,
_class"
 loc:@encoder_1/rnn/TensorArray
ľ
7encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3=encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterencoder_1/rnn/while/Identityencoder_1/rnn/while/Selectencoder_1/rnn/while/Identity_1*
T0*,
_class"
 loc:@encoder_1/rnn/TensorArray*
_output_shapes
: 
z
encoder_1/rnn/while/add/yConst^encoder_1/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
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

#encoder_1/rnn/while/NextIteration_1NextIteration7encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
}
#encoder_1/rnn/while/NextIteration_2NextIterationencoder_1/rnn/while/Select_1* 
_output_shapes
:
*
T0
}
#encoder_1/rnn/while/NextIteration_3NextIterationencoder_1/rnn/while/Select_2*
T0* 
_output_shapes
:

}
#encoder_1/rnn/while/NextIteration_4NextIterationencoder_1/rnn/while/Select_3* 
_output_shapes
:
*
T0
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
encoder_1/rnn/while/Exit_2Exitencoder_1/rnn/while/Switch_2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
encoder_1/rnn/while/Exit_3Exitencoder_1/rnn/while/Switch_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
encoder_1/rnn/while/Exit_4Exitencoder_1/rnn/while/Switch_4*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
encoder_1/rnn/while/Exit_5Exitencoder_1/rnn/while/Switch_5*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Â
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
:˙˙˙˙˙˙˙˙˙*,
_class"
 loc:@encoder_1/rnn/TensorArray*

Tidx0
§
2encoder_1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3encoder_1/rnn/TensorArray$encoder_1/rnn/TensorArrayStack/rangeencoder_1/rnn/while/Exit_1*,
_class"
 loc:@encoder_1/rnn/TensorArray*
element_shape:
*%
_output_shapes
:*
dtype0
q
encoder_1/rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
ł
encoder_1/rnn/transpose	Transpose2encoder_1/rnn/TensorArrayStack/TensorArrayGatherV3encoder_1/rnn/transpose/perm*
Tperm0*
T0*%
_output_shapes
:
Ż
6output_projection/W/Initializer/truncated_normal/shapeConst*&
_class
loc:@output_projection/W*
valueB"      *
_output_shapes
:*
dtype0
˘
5output_projection/W/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*&
_class
loc:@output_projection/W*
valueB
 *    
¤
7output_projection/W/Initializer/truncated_normal/stddevConst*&
_class
loc:@output_projection/W*
valueB
 *ÍĚĚ=*
_output_shapes
: *
dtype0

@output_projection/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6output_projection/W/Initializer/truncated_normal/shape*

seed *
T0*&
_class
loc:@output_projection/W*
seed2 *
dtype0*
_output_shapes
:	

4output_projection/W/Initializer/truncated_normal/mulMul@output_projection/W/Initializer/truncated_normal/TruncatedNormal7output_projection/W/Initializer/truncated_normal/stddev*
T0*
_output_shapes
:	*&
_class
loc:@output_projection/W
ö
0output_projection/W/Initializer/truncated_normalAdd4output_projection/W/Initializer/truncated_normal/mul5output_projection/W/Initializer/truncated_normal/mean*
T0*&
_class
loc:@output_projection/W*
_output_shapes
:	
ą
output_projection/W
VariableV2*
	container *
dtype0*&
_class
loc:@output_projection/W*
_output_shapes
:	*
shape:	*
shared_name 
ć
output_projection/W/AssignAssignoutput_projection/W0output_projection/W/Initializer/truncated_normal*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*&
_class
loc:@output_projection/W
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
valueB*ÍĚĚ=*
_output_shapes
:*
dtype0
§
output_projection/b
VariableV2*
_output_shapes
:*
dtype0*
shape:*
	container *&
_class
loc:@output_projection/b*
shared_name 
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
output_projection/b/readIdentityoutput_projection/b*
_output_shapes
:*&
_class
loc:@output_projection/b*
T0
ş
"output_projection/xw_plus_b/MatMulMatMulencoder_1/rnn/while/Exit_4output_projection/W/read*
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0
­
output_projection/xw_plus_bBiasAdd"output_projection/xw_plus_b/MatMuloutput_projection/b/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
output_projection/SoftmaxSoftmaxoutput_projection/xw_plus_b*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
"output_projection/ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0

output_projection/ArgMaxArgMaxoutput_projection/xw_plus_b"output_projection/ArgMax/dimension*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tidx0
o

loss/ConstConst*1
value(B&"  ?žz?`ĺp?bX?¤p}?  ?mç{>*
dtype0*
_output_shapes
:
j
loss/MulMuloutput_projection/xw_plus_b
loss/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
	loss/CastCastdec_targets*
_output_shapes
:	*

DstT0*

SrcT0
]
loss/logistic_loss/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
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
loss/logistic_loss/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
y
loss/logistic_loss/addAddloss/logistic_loss/add/xloss/logistic_loss/mul*
T0*
_output_shapes
:	
_
loss/logistic_loss/sub_1/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
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
loss/logistic_loss/AbsAbsloss/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
loss/logistic_loss/NegNegloss/logistic_loss/Abs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
loss/logistic_loss/ExpExploss/logistic_loss/Neg*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
loss/logistic_loss/Log1pLog1ploss/logistic_loss/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
[
loss/logistic_loss/Neg_1Negloss/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
loss/logistic_loss/ReluReluloss/logistic_loss/Neg_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

loss/logistic_loss/add_1Addloss/logistic_loss/Log1ploss/logistic_loss/Relu*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
valueB"       *
_output_shapes
:*
dtype0
o
loss/SumSumloss/logistic_lossloss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
]
loss/Const_2Const*
valueB"       *
_output_shapes
:*
dtype0
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
s
accuracy/ArgMaxArgMaxdec_targetsaccuracy/ArgMax/dimension*

Tidx0*
T0*
_output_shapes	
:
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
accuracy/accuracyMeanaccuracy/Castaccuracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
`
learning_rate/learning_rateConst*
valueB
 *ˇQ8*
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
learning_rate/Cast_1/xConst*
value
B :'*
dtype0*
_output_shapes
: 
d
learning_rate/Cast_1Castlearning_rate/Cast_1/x*
_output_shapes
: *

DstT0*

SrcT0
[
learning_rate/Cast_2/xConst*
valueB
 *Âu?*
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
gradients/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
b
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
:	*
T0
S
gradients/f_countConst*
_output_shapes
: *
dtype0*
value	B : 
¸
gradients/f_count_1Entergradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
N*
T0*
_output_shapes
: : 
l
gradients/SwitchSwitchgradients/Mergeencoder_1/rnn/while/LoopCond*
T0*
_output_shapes
: : 
p
gradients/Add/yConst^encoder_1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
_output_shapes
: *
T0
š
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
gradients/b_countConst*
_output_shapes
: *
dtype0*
value	B :
Ä
gradients/b_count_1Entergradients/f_count_2*
is_constant( *
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
Ë
gradients/GreaterEqual/EnterEntergradients/b_count*
parallel_iterations *
T0*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(
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
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 

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
valueB"      
z
)gradients/loss/logistic_loss_grad/Shape_1Const*
valueB"      *
_output_shapes
:*
dtype0
á
7gradients/loss/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/loss/logistic_loss_grad/Shape)gradients/loss/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ľ
%gradients/loss/logistic_loss_grad/SumSumgradients/Fill7gradients/loss/logistic_loss_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ź
)gradients/loss/logistic_loss_grad/ReshapeReshape%gradients/loss/logistic_loss_grad/Sum'gradients/loss/logistic_loss_grad/Shape*
_output_shapes
:	*
Tshape0*
T0
š
'gradients/loss/logistic_loss_grad/Sum_1Sumgradients/Fill9gradients/loss/logistic_loss_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Â
+gradients/loss/logistic_loss_grad/Reshape_1Reshape'gradients/loss/logistic_loss_grad/Sum_1)gradients/loss/logistic_loss_grad/Shape_1*
_output_shapes
:	*
Tshape0*
T0
~
-gradients/loss/logistic_loss/mul_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      
w
/gradients/loss/logistic_loss/mul_1_grad/Shape_1Shapeloss/Mul*
out_type0*
_output_shapes
:*
T0
ó
=gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/mul_1_grad/Shape/gradients/loss/logistic_loss/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

+gradients/loss/logistic_loss/mul_1_grad/mulMul)gradients/loss/logistic_loss_grad/Reshapeloss/Mul*
T0*
_output_shapes
:	
Ţ
+gradients/loss/logistic_loss/mul_1_grad/SumSum+gradients/loss/logistic_loss/mul_1_grad/mul=gradients/loss/logistic_loss/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Î
/gradients/loss/logistic_loss/mul_1_grad/ReshapeReshape+gradients/loss/logistic_loss/mul_1_grad/Sum-gradients/loss/logistic_loss/mul_1_grad/Shape*
T0*
_output_shapes
:	*
Tshape0
Ł
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
1gradients/loss/logistic_loss/mul_1_grad/Reshape_1Reshape-gradients/loss/logistic_loss/mul_1_grad/Sum_1/gradients/loss/logistic_loss/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
-gradients/loss/logistic_loss/mul_2_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
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
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ł
+gradients/loss/logistic_loss/mul_2_grad/mulMul+gradients/loss/logistic_loss_grad/Reshape_1loss/logistic_loss/add_1*
T0*
_output_shapes
:	
Ţ
+gradients/loss/logistic_loss/mul_2_grad/SumSum+gradients/loss/logistic_loss/mul_2_grad/mul=gradients/loss/logistic_loss/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Î
/gradients/loss/logistic_loss/mul_2_grad/ReshapeReshape+gradients/loss/logistic_loss/mul_2_grad/Sum-gradients/loss/logistic_loss/mul_2_grad/Shape*
_output_shapes
:	*
Tshape0*
T0
Ł
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
1gradients/loss/logistic_loss/mul_2_grad/Reshape_1Reshape-gradients/loss/logistic_loss/mul_2_grad/Sum_1/gradients/loss/logistic_loss/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

-gradients/loss/logistic_loss/add_1_grad/ShapeShapeloss/logistic_loss/Log1p*
out_type0*
_output_shapes
:*
T0

/gradients/loss/logistic_loss/add_1_grad/Shape_1Shapeloss/logistic_loss/Relu*
T0*
out_type0*
_output_shapes
:
ó
=gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/loss/logistic_loss/add_1_grad/Shape/gradients/loss/logistic_loss/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ä
+gradients/loss/logistic_loss/add_1_grad/SumSum1gradients/loss/logistic_loss/mul_2_grad/Reshape_1=gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ö
/gradients/loss/logistic_loss/add_1_grad/ReshapeReshape+gradients/loss/logistic_loss/add_1_grad/Sum-gradients/loss/logistic_loss/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
-gradients/loss/logistic_loss/add_1_grad/Sum_1Sum1gradients/loss/logistic_loss/mul_2_grad/Reshape_1?gradients/loss/logistic_loss/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ü
1gradients/loss/logistic_loss/add_1_grad/Reshape_1Reshape-gradients/loss/logistic_loss/add_1_grad/Sum_1/gradients/loss/logistic_loss/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
-gradients/loss/logistic_loss/Log1p_grad/add/xConst0^gradients/loss/logistic_loss/add_1_grad/Reshape*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ť
+gradients/loss/logistic_loss/Log1p_grad/addAdd-gradients/loss/logistic_loss/Log1p_grad/add/xloss/logistic_loss/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

2gradients/loss/logistic_loss/Log1p_grad/Reciprocal
Reciprocal+gradients/loss/logistic_loss/Log1p_grad/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
+gradients/loss/logistic_loss/Log1p_grad/mulMul/gradients/loss/logistic_loss/add_1_grad/Reshape2gradients/loss/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
š
/gradients/loss/logistic_loss/Relu_grad/ReluGradReluGrad1gradients/loss/logistic_loss/add_1_grad/Reshape_1loss/logistic_loss/Relu*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
§
)gradients/loss/logistic_loss/Exp_grad/mulMul+gradients/loss/logistic_loss/Log1p_grad/mulloss/logistic_loss/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

+gradients/loss/logistic_loss/Neg_1_grad/NegNeg/gradients/loss/logistic_loss/Relu_grad/ReluGrad*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

)gradients/loss/logistic_loss/Neg_grad/NegNeg)gradients/loss/logistic_loss/Exp_grad/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
n
*gradients/loss/logistic_loss/Abs_grad/SignSignloss/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
)gradients/loss/logistic_loss/Abs_grad/mulMul)gradients/loss/logistic_loss/Neg_grad/Neg*gradients/loss/logistic_loss/Abs_grad/Sign*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
gradients/AddNAddN1gradients/loss/logistic_loss/mul_1_grad/Reshape_1+gradients/loss/logistic_loss/Neg_1_grad/Neg)gradients/loss/logistic_loss/Abs_grad/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N*D
_class:
86loc:@gradients/loss/logistic_loss/mul_1_grad/Reshape_1*
T0
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
Ă
-gradients/loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/Mul_grad/Shapegradients/loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
p
gradients/loss/Mul_grad/mulMulgradients/AddN
loss/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
gradients/loss/Mul_grad/SumSumgradients/loss/Mul_grad/mul-gradients/loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ś
gradients/loss/Mul_grad/ReshapeReshapegradients/loss/Mul_grad/Sumgradients/loss/Mul_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/loss/Mul_grad/mul_1Muloutput_projection/xw_plus_bgradients/AddN*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
´
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
˘
6gradients/output_projection/xw_plus_b_grad/BiasAddGradBiasAddGradgradients/loss/Mul_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
Ö
8gradients/output_projection/xw_plus_b/MatMul_grad/MatMulMatMulgradients/loss/Mul_grad/Reshapeoutput_projection/W/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
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
:˙˙˙˙˙˙˙˙˙
r
gradients/zeros_like_2	ZerosLikeencoder_1/rnn/while/Exit_3*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
gradients/zeros_like_3	ZerosLikeencoder_1/rnn/while/Exit_5*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

0gradients/encoder_1/rnn/while/Exit_4_grad/b_exitEnter8gradients/output_projection/xw_plus_b/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
ä
0gradients/encoder_1/rnn/while/Exit_1_grad/b_exitEntergradients/zeros_like*
parallel_iterations *
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
ö
0gradients/encoder_1/rnn/while/Exit_2_grad/b_exitEntergradients/zeros_like_1*
parallel_iterations *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
ö
0gradients/encoder_1/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_like_2*
is_constant( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
ö
0gradients/encoder_1/rnn/while/Exit_5_grad/b_exitEntergradients/zeros_like_3*
is_constant( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
parallel_iterations 
ę
4gradients/encoder_1/rnn/while/Switch_4_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_4_grad/b_exit;gradients/encoder_1/rnn/while/Switch_4_grad_1/NextIteration**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
N*
T0
ę
4gradients/encoder_1/rnn/while/Switch_2_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_2_grad/b_exit;gradients/encoder_1/rnn/while/Switch_2_grad_1/NextIteration**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N
ę
4gradients/encoder_1/rnn/while/Switch_3_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_3_grad/b_exit;gradients/encoder_1/rnn/while/Switch_3_grad_1/NextIteration**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N
ę
4gradients/encoder_1/rnn/while/Switch_5_grad/b_switchMerge0gradients/encoder_1/rnn/while/Exit_5_grad/b_exit;gradients/encoder_1/rnn/while/Switch_5_grad_1/NextIteration**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
N*
T0

1gradients/encoder_1/rnn/while/Merge_4_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_4_grad/b_switch*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙:
*
T0

1gradients/encoder_1/rnn/while/Merge_2_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*
T0*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_2_grad/b_switch*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙:


1gradients/encoder_1/rnn/while/Merge_3_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙:
*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_3_grad/b_switch

1gradients/encoder_1/rnn/while/Merge_5_grad/SwitchSwitch4gradients/encoder_1/rnn/while/Switch_5_grad/b_switchgradients/b_count_2*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Switch_5_grad/b_switch*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙:
*
T0

/gradients/encoder_1/rnn/while/Enter_4_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_4_grad/Switch*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/encoder_1/rnn/while/Enter_2_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_2_grad/Switch*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/encoder_1/rnn/while/Enter_3_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_3_grad/Switch*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/encoder_1/rnn/while/Enter_5_grad/ExitExit1gradients/encoder_1/rnn/while/Merge_5_grad/Switch*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

4gradients/encoder_1/initial_state_2_tiled_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
Ű
4gradients/encoder_1/initial_state_2_tiled_grad/stackPack)encoder_1/initial_state_2_tiled/multiples4gradients/encoder_1/initial_state_2_tiled_grad/Shape*
T0*

axis *
N*
_output_shapes

:

=gradients/encoder_1/initial_state_2_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_2_tiled_grad/stack*
_output_shapes
: *
T0

>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub/yConst*
dtype0*
_output_shapes
: *
value	B :
ă
<gradients/encoder_1/initial_state_2_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_2_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_2_tiled_grad/transpose/sub/y*
_output_shapes
: *
T0

Dgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 

Dgradients/encoder_1/initial_state_2_tiled_grad/transpose/Range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
ş
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
Tperm0*
T0*
_output_shapes

:

<gradients/encoder_1/initial_state_2_tiled_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
ě
6gradients/encoder_1/initial_state_2_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_2_tiled_grad/transpose<gradients/encoder_1/initial_state_2_tiled_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
u
3gradients/encoder_1/initial_state_2_tiled_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
|
:gradients/encoder_1/initial_state_2_tiled_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
|
:gradients/encoder_1/initial_state_2_tiled_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0

4gradients/encoder_1/initial_state_2_tiled_grad/rangeRange:gradients/encoder_1/initial_state_2_tiled_grad/range/start3gradients/encoder_1/initial_state_2_tiled_grad/Size:gradients/encoder_1/initial_state_2_tiled_grad/range/delta*

Tidx0*
_output_shapes
:

8gradients/encoder_1/initial_state_2_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_4_grad/Exit6gradients/encoder_1/initial_state_2_tiled_grad/Reshape*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
đ
2gradients/encoder_1/initial_state_2_tiled_grad/SumSum8gradients/encoder_1/initial_state_2_tiled_grad/Reshape_14gradients/encoder_1/initial_state_2_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	
ˇ
<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_accStack*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*

stack_name *
_output_shapes
:
É
?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
§
@gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_4^gradients/Add*
_output_shapes
:*
swap_memory( *1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
T0
Ü
Hgradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/

?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@encoder_1/rnn/while/Identity_4

=gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/b_syncControlTrigger@^gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop@^gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop<^gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopa^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopd^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopV^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopX^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopY^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPopf^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopa^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopd^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopb^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop
ˇ
6gradients/encoder_1/rnn/while/Select_3_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ˇ
8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*

stack_name *
_output_shapes
:
Ĺ
;gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc*
T0*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
§
<gradients/encoder_1/rnn/while/Select_3_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_3^gradients/Add*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
_output_shapes
:*
swap_memory( *
T0

Ř
Dgradients/encoder_1/rnn/while/Select_3_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_3*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
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
Ű
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
>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub/yConst*
dtype0*
_output_shapes
: *
value	B :
ă
<gradients/encoder_1/initial_state_0_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_0_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_0_tiled_grad/transpose/sub/y*
_output_shapes
: *
T0

Dgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/startConst*
_output_shapes
: *
dtype0*
value	B : 

Dgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
ş
>gradients/encoder_1/initial_state_0_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_0_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_0_tiled_grad/transpose/Range/delta*

Tidx0*
_output_shapes
:
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
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
ě
6gradients/encoder_1/initial_state_0_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_0_tiled_grad/transpose<gradients/encoder_1/initial_state_0_tiled_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
u
3gradients/encoder_1/initial_state_0_tiled_grad/SizeConst*
value	B :*
_output_shapes
: *
dtype0
|
:gradients/encoder_1/initial_state_0_tiled_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
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
8gradients/encoder_1/initial_state_0_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_2_grad/Exit6gradients/encoder_1/initial_state_0_tiled_grad/Reshape*
Tshape0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
đ
2gradients/encoder_1/initial_state_0_tiled_grad/SumSum8gradients/encoder_1/initial_state_0_tiled_grad/Reshape_14gradients/encoder_1/initial_state_0_tiled_grad/range*
_output_shapes
:	*
T0*
	keep_dims( *

Tidx0
ˇ
<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_accStack*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*

stack_name *
_output_shapes
:
É
?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
§
@gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPush	StackPush?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnterencoder_1/rnn/while/Identity_2^gradients/Add*
T0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*
_output_shapes
:*
swap_memory( 
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
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
6gradients/encoder_1/rnn/while/Select_1_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_accStack*
	elem_type0
*

stack_name *
_output_shapes
:*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1
Ĺ
;gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1
§
<gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_1^gradients/Add*
_output_shapes
:*
swap_memory( *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1*
T0

Ř
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
*
_output_shapes	
:*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_1

2gradients/encoder_1/rnn/while/Select_1_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_1_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_2_grad/Switch:16gradients/encoder_1/rnn/while/Select_1_grad/zeros_like*
T0* 
_output_shapes
:

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
Ű
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
value	B :*
dtype0*
_output_shapes
: 
ă
<gradients/encoder_1/initial_state_1_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_1_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_1_tiled_grad/transpose/sub/y*
T0*
_output_shapes
: 

Dgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/startConst*
value	B : *
_output_shapes
: *
dtype0

Dgradients/encoder_1/initial_state_1_tiled_grad/transpose/Range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
ş
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
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
ě
6gradients/encoder_1/initial_state_1_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_1_tiled_grad/transpose<gradients/encoder_1/initial_state_1_tiled_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
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
:gradients/encoder_1/initial_state_1_tiled_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :

4gradients/encoder_1/initial_state_1_tiled_grad/rangeRange:gradients/encoder_1/initial_state_1_tiled_grad/range/start3gradients/encoder_1/initial_state_1_tiled_grad/Size:gradients/encoder_1/initial_state_1_tiled_grad/range/delta*
_output_shapes
:*

Tidx0

8gradients/encoder_1/initial_state_1_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_3_grad/Exit6gradients/encoder_1/initial_state_1_tiled_grad/Reshape*
T0*
Tshape0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
đ
2gradients/encoder_1/initial_state_1_tiled_grad/SumSum8gradients/encoder_1/initial_state_1_tiled_grad/Reshape_14gradients/encoder_1/initial_state_1_tiled_grad/range*
_output_shapes
:	*
T0*
	keep_dims( *

Tidx0
ˇ
<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3
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
Hgradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop/RefEnterRefEnter<gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*
T0

?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPopStackPopHgradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop/RefEnter^gradients/Sub*
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_3*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
6gradients/encoder_1/rnn/while/Select_2_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_accStack*
	elem_type0
*
_output_shapes
:*

stack_name *5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2
Ĺ
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
<gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_2^gradients/Add*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
_output_shapes
:*
swap_memory( *
T0

Ř
Dgradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc*
is_constant(*
T0*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 

;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_2*
_output_shapes	
:

2gradients/encoder_1/rnn/while/Select_2_grad/SelectSelect;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop3gradients/encoder_1/rnn/while/Merge_3_grad/Switch:16gradients/encoder_1/rnn/while/Select_2_grad/zeros_like*
T0* 
_output_shapes
:


4gradients/encoder_1/rnn/while/Select_2_grad/Select_1Select;gradients/encoder_1/rnn/while/Select_2_grad/Select/StackPop6gradients/encoder_1/rnn/while/Select_2_grad/zeros_like3gradients/encoder_1/rnn/while/Merge_3_grad/Switch:1*
T0* 
_output_shapes
:


4gradients/encoder_1/initial_state_3_tiled_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ű
4gradients/encoder_1/initial_state_3_tiled_grad/stackPack)encoder_1/initial_state_3_tiled/multiples4gradients/encoder_1/initial_state_3_tiled_grad/Shape*

axis *
_output_shapes

:*
T0*
N

=gradients/encoder_1/initial_state_3_tiled_grad/transpose/RankRank4gradients/encoder_1/initial_state_3_tiled_grad/stack*
T0*
_output_shapes
: 

>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
ă
<gradients/encoder_1/initial_state_3_tiled_grad/transpose/subSub=gradients/encoder_1/initial_state_3_tiled_grad/transpose/Rank>gradients/encoder_1/initial_state_3_tiled_grad/transpose/sub/y*
_output_shapes
: *
T0

Dgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/startConst*
_output_shapes
: *
dtype0*
value	B : 

Dgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
ş
>gradients/encoder_1/initial_state_3_tiled_grad/transpose/RangeRangeDgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/start=gradients/encoder_1/initial_state_3_tiled_grad/transpose/RankDgradients/encoder_1/initial_state_3_tiled_grad/transpose/Range/delta*

Tidx0*
_output_shapes
:
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
<gradients/encoder_1/initial_state_3_tiled_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
ě
6gradients/encoder_1/initial_state_3_tiled_grad/ReshapeReshape8gradients/encoder_1/initial_state_3_tiled_grad/transpose<gradients/encoder_1/initial_state_3_tiled_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
u
3gradients/encoder_1/initial_state_3_tiled_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
|
:gradients/encoder_1/initial_state_3_tiled_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
|
:gradients/encoder_1/initial_state_3_tiled_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

4gradients/encoder_1/initial_state_3_tiled_grad/rangeRange:gradients/encoder_1/initial_state_3_tiled_grad/range/start3gradients/encoder_1/initial_state_3_tiled_grad/Size:gradients/encoder_1/initial_state_3_tiled_grad/range/delta*
_output_shapes
:*

Tidx0

8gradients/encoder_1/initial_state_3_tiled_grad/Reshape_1Reshape/gradients/encoder_1/rnn/while/Enter_5_grad/Exit6gradients/encoder_1/initial_state_3_tiled_grad/Reshape*
Tshape0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
đ
2gradients/encoder_1/initial_state_3_tiled_grad/SumSum8gradients/encoder_1/initial_state_3_tiled_grad/Reshape_14gradients/encoder_1/initial_state_3_tiled_grad/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	
ˇ
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
	elem_type0*1
_class'
%#loc:@encoder_1/rnn/while/Identity_5*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
6gradients/encoder_1/rnn/while/Select_4_grad/zeros_like	ZerosLike?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ˇ
8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_accStack*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
_output_shapes
:*

stack_name 
Ĺ
;gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4
§
<gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPush	StackPush;gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnter"encoder_1/rnn/while/GreaterEqual_4^gradients/Add*
T0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
_output_shapes
:*
swap_memory( 
Ř
Dgradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop/RefEnterRefEnter8gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4

;gradients/encoder_1/rnn/while/Select_4_grad/Select/StackPopStackPopDgradients/encoder_1/rnn/while/Select_4_grad/Select/StackPop/RefEnter^gradients/Sub*
	elem_type0
*5
_class+
)'loc:@encoder_1/rnn/while/GreaterEqual_4*
_output_shapes	
:
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
Ż
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
ą
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
valueB"      *
_output_shapes
:*
dtype0
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
é
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_accStack*
	elem_type0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc*
is_constant(*
T0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnter:encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1^gradients/Add*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
_output_shapes
:*
swap_memory( *
T0
¤
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
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
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mulMul4gradients/encoder_1/rnn/while/Select_4_grad/Select_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPop*
T0* 
_output_shapes
:

Á
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
˛
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape* 
_output_shapes
:
*
Tshape0*
T0
î
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_accStack*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2^gradients/Add*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
_output_shapes
:*
swap_memory( *
T0
Ť
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2
Ú
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
Ç
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¸
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*
T0*
Tshape0* 
_output_shapes
:

˝
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape* 
_output_shapes
:
*
T0
´
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1* 
_output_shapes
:
*
T0

gradients/AddN_1AddN4gradients/encoder_1/rnn/while/Select_3_grad/Select_1Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGrad*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_3_grad/Select_1* 
_output_shapes
:
*
T0*
N
Ż
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
ą
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/SumSumgradients/AddN_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
˛
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
¸
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1*
T0*
Tshape0* 
_output_shapes
:

­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
Ź
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
Ĺ
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/
Ó
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1
Ř
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
T0

egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
:*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1
ç
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shapeegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mulMulPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape?gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/StackPop* 
_output_shapes
:
*
T0
ť
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumSumJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ź
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape*
T0*
Tshape0* 
_output_shapes
:

ę
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_accStack*
	elem_type0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*

stack_name *
_output_shapes
:
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
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnter;encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid^gradients/Add*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
_output_shapes
:*
swap_memory( *
T0
Ľ
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
Ô
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid
§
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1MulUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape*
T0* 
_output_shapes
:

Á
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ń
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ż
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
ą
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ç
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_accStack*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
_output_shapes
:*

stack_name 

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnter8encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh
˘
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0*
is_constant(
Ń
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh* 
_output_shapes
:

Š
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mulMulRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPop*
T0* 
_output_shapes
:

Á
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
˛
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape*
T0* 
_output_shapes
:
*
Tshape0
î
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1^gradients/Add*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
_output_shapes
:*
swap_memory( 
Ť
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc*
is_constant(*
T0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
Ú
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1
­
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1* 
_output_shapes
:
*
T0
Ç
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¸
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1*
T0*
Tshape0* 
_output_shapes
:

ˇ
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
˝
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape*
T0* 
_output_shapes
:

˛
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1* 
_output_shapes
:
*
T0
­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
Ą
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
valueB *
_output_shapes
: *
dtype0
Đ
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ç
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumSumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ź
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0
Ë
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1SumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¨
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
ő
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_accStack*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*

stack_name *
_output_shapes
:
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
Ł
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPush	StackPushXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnterCencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim^gradients/Add*
_output_shapes
:*
swap_memory( *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
T0
ł
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
Ř
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPopStackPopagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
: *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split/split_dim
Ë
Ogradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concatConcatV2Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/StackPop*

Tidx0*
T0*
N* 
_output_shapes
:

ó
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:
Ŕ
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
	elem_type0*
_output_shapes
:*

stack_name *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat
ť
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
T0
ż
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterDencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat^gradients/Add*
T0*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
_output_shapes
:*
swap_memory( 
Î
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
ý
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat
ď
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1MatMulegradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
transpose_b( * 
_output_shapes
:
*
transpose_a(*
T0
Ľ
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
Ě
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:: 
ý
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
::
´
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/AddAddYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch:1Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
ë
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
ß
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:
Ş
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/RankConst^gradients/Sub*
_output_shapes
: *
dtype0*
value	B :

]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_accStack*
	elem_type0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*

stack_name *
_output_shapes
:
ś
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis
ż
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPush	StackPush`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnterIencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis^gradients/Add*
T0*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*
swap_memory( 
É
igradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis*
T0*
is_constant(
î
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopStackPopigradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
: *\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat/axis
Ŕ
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/modFloorMod`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/StackPopXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
ş
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
š
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Shape_1Shapeencoder_1/rnn/while/Identity_5*
T0*
out_type0*
_output_shapes
:
ö
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2
Ź
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
T0*
is_constant(
Ľ
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPush	StackPushcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter9encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2^gradients/Add*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:*
swap_memory( *
T0
ż
lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
î
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopStackPoplgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*L
_classB
@>loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2
Î
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeNShapeNcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop?gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/StackPop*
out_type0* 
_output_shapes
::*
T0*
N
Ž
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffsetConcatOffsetWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/modZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
´
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/SliceSliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffsetZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN*
Index0*
T0* 
_output_shapes
:

Â
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice_1SliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMulbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ConcatOffset:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_accConst*
dtype0* 
_output_shapes
:
*
valueB
*    
č
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1Enter_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc*
parallel_iterations *
T0* 
_output_shapes
:
*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant( 
ě
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2Mergeagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIteration*
N*
T0*"
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
Ś
gradients/AddN_3AddN4gradients/encoder_1/rnn/while/Select_2_grad/Select_1Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_2_grad/Select_1* 
_output_shapes
:
*
T0*
N
Ż
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
ą
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
é
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_accStack*
	elem_type0*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnter:encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1^gradients/Add*M
_classC
A?loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
_output_shapes
:*
swap_memory( *
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
ç
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mulMulgradients/AddN_3Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPop* 
_output_shapes
:
*
T0
Á
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
˛
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0
î
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_accStack*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2^gradients/Add*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
_output_shapes
:*
swap_memory( *
T0
Ť
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
Ú
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopStackPop`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2* 
_output_shapes
:

ë
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1MulWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopgradients/AddN_3* 
_output_shapes
:
*
T0
Ç
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¸
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*
Tshape0* 
_output_shapes
:
*
T0
¤
gradients/AddN_4AddN2gradients/encoder_1/rnn/while/Select_4_grad/Select[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/Slice_1*
N*
T0* 
_output_shapes
:
*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_4_grad/Select
˝
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape*
T0* 
_output_shapes
:

´
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGradTanhGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/StackPopRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1*
T0* 
_output_shapes
:


;gradients/encoder_1/rnn/while/Switch_5_grad_1/NextIterationNextIterationgradients/AddN_4* 
_output_shapes
:
*
T0

gradients/AddN_5AddN4gradients/encoder_1/rnn/while/Select_1_grad/Select_1Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGrad*G
_class=
;9loc:@gradients/encoder_1/rnn/while/Select_1_grad/Select_1* 
_output_shapes
:
*
T0*
N
Ż
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
ą
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/SumSumgradients/AddN_5^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
˛
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape* 
_output_shapes
:
*
Tshape0*
T0

Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_5`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¸
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*
T0* 
_output_shapes
:
*
Tshape0
­
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
Ź
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1Shapeencoder_1/rnn/while/Identity_2*
out_type0*
_output_shapes
:*
T0

bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStack*
	elem_type0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*

stack_name 
Ĺ
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
is_constant(*
T0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
Ó
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnterNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1^gradients/Add*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:*
swap_memory( *
T0
Ř
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
	elem_type0*a
_classW
USloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
_output_shapes
:
ç
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shapeegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mulMulPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape?gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/StackPop*
T0* 
_output_shapes
:

ť
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumSumJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ź
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape*
Tshape0* 
_output_shapes
:
*
T0
ę
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
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnter;encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid^gradients/Add*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
_output_shapes
:*
swap_memory( *
T0
Ľ
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc*
is_constant(*
T0*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
Ô
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*N
_classD
B@loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid
§
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1MulUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape* 
_output_shapes
:
*
T0
Á
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ń
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPop*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
Ż
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
ą
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
Ö
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapePgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ç
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_accStack*
	elem_type0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*

stack_name *
_output_shapes
:

Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh

Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPush	StackPushUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnter8encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh^gradients/Add*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
_output_shapes
:*
swap_memory( *
T0
˘
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop/RefEnterRefEnterRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc*
is_constant(*
T0*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations 
Ń
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPopStackPop^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*K
_classA
?=loc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh
Š
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mulMulRgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/StackPop* 
_output_shapes
:
*
T0
Á
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumSumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
˛
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ReshapeReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape* 
_output_shapes
:
*
Tshape0*
T0
î
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1

Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1

Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPush	StackPushWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnter=encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1^gradients/Add*P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
_output_shapes
:*
swap_memory( *
T0
Ť
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPop/RefEnterRefEnterTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *P
_classF
DBloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
T0
Ú
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
Ç
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1SumNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¸
Rgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1ReshapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*
T0*
Tshape0* 
_output_shapes
:

ˇ
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/StackPopNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape* 
_output_shapes
:
*
T0

gradients/AddN_6AddN2gradients/encoder_1/rnn/while/Select_1_grad/SelectPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_1_grad/Select* 
_output_shapes
:
*
T0*
N
˝
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/StackPopPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape* 
_output_shapes
:
*
T0
˛
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
Ą
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
_output_shapes
: *
dtype0*
valueB 
Đ
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ç
Jgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumSumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ź
Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeReshapeJgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape* 
_output_shapes
:
*
Tshape0*
T0
Ë
Lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1SumVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¨
Pgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1ReshapeLgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1Ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

;gradients/encoder_1/rnn/while/Switch_2_grad_1/NextIterationNextIterationgradients/AddN_6* 
_output_shapes
:
*
T0
ő
Ugradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim
 
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
T0
Ł
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPush	StackPushXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnterCencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim^gradients/Add*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
_output_shapes
:*
swap_memory( *
T0
ł
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop/RefEnterRefEnterUgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
Ř
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPopStackPopagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop/RefEnter^gradients/Sub*
	elem_type0*V
_classL
JHloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split/split_dim*
_output_shapes
: 
Ë
Ogradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concatConcatV2Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradPgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradNgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/StackPop* 
_output_shapes
:
*
T0*

Tidx0*
N
ó
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
_output_shapes	
:*
data_formatNHWC*
T0
Ŕ
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
	elem_type0*
_output_shapes
:*

stack_name *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
ť
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat*
T0*
is_constant(
ż
fgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPush	StackPushegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnterDencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat^gradients/Add*
T0*
_output_shapes
:*
swap_memory( *W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
Î
ngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnterbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
ý
egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopStackPopngradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*
	elem_type0* 
_output_shapes
:
*W
_classM
KIloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat
ď
\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1MatMulegradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/StackPopOgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
Ľ
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
_output_shapes	
:*
valueB*    
Ń
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterVgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc*
_output_shapes	
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant( *
T0
Ě
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIteration*
_output_shapes
	:: *
T0*
N
ý
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*"
_output_shapes
::*
T0
´
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/AddAddYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch:1Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
ë
^gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationTgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
ß
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:
Ş
Xgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
_output_shapes
: *
dtype0

]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_accStack*
	elem_type0*

stack_name *
_output_shapes
:*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis
ś
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
is_constant(*
parallel_iterations *\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
T0
ż
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPush	StackPush`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnterIencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis^gradients/Add*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*
swap_memory( *
T0
É
igradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnterRefEnter]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc*\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
î
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopStackPopigradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPop/RefEnter^gradients/Sub*
	elem_type0*
_output_shapes
: *\
_classR
PNloc:@encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat/axis
Ŕ
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/modFloorMod`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/StackPopXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
ş
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
š
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Shape_1Shapeencoder_1/rnn/while/Identity_3*
out_type0*
_output_shapes
:*
T0
Ř
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_accStack*
	elem_type0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
:*

stack_name 

cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*
parallel_iterations *
is_constant(*
T0
ó
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPush	StackPushcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter%encoder_1/rnn/while/TensorArrayReadV3^gradients/Add*
T0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
:*
swap_memory( 
Ą
lgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnterRefEnter`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
Đ
cgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPopStackPoplgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop/RefEnter^gradients/Sub*
	elem_type0*.
_class$
" loc:@encoder_1/rnn/TensorArray_1* 
_output_shapes
:

Î
Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeNShapeNcgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/StackPop?gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/StackPop* 
_output_shapes
::*
N*
out_type0*
T0
Ž
`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffsetConcatOffsetWgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/modZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
´
Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/SliceSliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffsetZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN*
Index0*
T0* 
_output_shapes
:

Â
[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Slice_1SliceZgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMulbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ConcatOffset:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Index0*
T0
¸
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
ě
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2Mergeagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_1ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
: 

`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/SwitchSwitchagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_output_shapes
:
:
*
T0
Ń
]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/AddAddbgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch:1\gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0

ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/NextIterationNextIteration]gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:

ö
agradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3Exit`gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:

É
\gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterencoder_1/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
ô
^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterHencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/*
T0*
is_constant(*
parallel_iterations 
 
Vgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3\gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*
_output_shapes

::*
source	gradients*.
_class$
" loc:@encoder_1/rnn/TensorArray_1
č
Rgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1W^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: *
T0
ů
^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_accStack*
	elem_type0*
_output_shapes
:*

stack_name *Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity
­
agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnterRefEnter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*8

frame_name*(encoder_1/rnn/while/encoder_1/rnn/while/*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity

bgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPush	StackPushagradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnterencoder_1/rnn/while/Identity^gradients/Add*
_output_shapes
:*
swap_memory( *Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
T0
Ŕ
jgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop/RefEnterRefEnter^gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*
T0*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/
ĺ
agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopStackPopjgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPop/RefEnter^gradients/Sub*
	elem_type0*Q
_classG
E loc:@encoder_1/rnn/TensorArray_1!loc:@encoder_1/rnn/while/Identity*
_output_shapes
: 
Š
Xgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Vgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3agradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopYgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/SliceRgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes
: *
T0
¤
gradients/AddN_7AddN2gradients/encoder_1/rnn/while/Select_2_grad/Select[gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/Slice_1*
N*
T0* 
_output_shapes
:
*E
_class;
97loc:@gradients/encoder_1/rnn/while/Select_2_grad/Select

Bgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterBgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *B

frame_name42gradients/encoder_1/rnn/while/encoder_1/rnn/while/

Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Jgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
_output_shapes
: : *
T0*
N
Ë
Cgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*
_output_shapes
: : *
T0

@gradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddEgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Xgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
ž
Jgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration@gradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
˛
Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitCgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 

;gradients/encoder_1/rnn/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_7*
T0* 
_output_shapes
:

Ř
ygradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3encoder_1/rnn/TensorArray_1Dgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
source	gradients*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
_output_shapes

::

ugradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityDgradients/encoder_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3z^gradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
T0

kgradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3ygradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3&encoder_1/rnn/TensorArrayUnstack/rangeugradients/encoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*.
_class$
" loc:@encoder_1/rnn/TensorArray_1*
element_shape:*%
_output_shapes
:*
dtype0
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
+gradients/encoder/conv1d/Squeeze_grad/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
Ő
-gradients/encoder/conv1d/Squeeze_grad/ReshapeReshape,gradients/encoder_1/transpose_grad/transpose+gradients/encoder/conv1d/Squeeze_grad/Shape*
Tshape0*)
_output_shapes
:*
T0

*gradients/encoder/conv1d/Conv2D_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
Ň
8gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/encoder/conv1d/Conv2D_grad/Shapeencoder/conv1d/ExpandDims_1-gradients/encoder/conv1d/Squeeze_grad/Reshape*
paddingVALID*
T0*
data_formatNHWC*
strides
*(
_output_shapes
:*
use_cudnn_on_gpu(

,gradients/encoder/conv1d/Conv2D_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:
Ó
9gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/conv1d/ExpandDims,gradients/encoder/conv1d/Conv2D_grad/Shape_1-gradients/encoder/conv1d/Squeeze_grad/Reshape*
data_formatNHWC*
strides
*'
_output_shapes
:*
paddingVALID*
T0*
use_cudnn_on_gpu(

0gradients/encoder/conv1d/ExpandDims_1_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         
ć
2gradients/encoder/conv1d/ExpandDims_1_grad/ReshapeReshape9gradients/encoder/conv1d/Conv2D_grad/Conv2DBackpropFilter0gradients/encoder/conv1d/ExpandDims_1_grad/Shape*
Tshape0*#
_output_shapes
:*
T0
¸
global_norm/L2LossL2Loss2gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*
_output_shapes
: *
T0
ş
global_norm/L2Loss_1L2Loss2gradients/encoder_1/initial_state_0_tiled_grad/Sum*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
: 
ş
global_norm/L2Loss_2L2Loss2gradients/encoder_1/initial_state_1_tiled_grad/Sum*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
_output_shapes
: *
T0
ş
global_norm/L2Loss_3L2Loss2gradients/encoder_1/initial_state_2_tiled_grad/Sum*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum*
_output_shapes
: *
T0
ş
global_norm/L2Loss_4L2Loss2gradients/encoder_1/initial_state_3_tiled_grad/Sum*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
_output_shapes
: 

global_norm/L2Loss_5L2Lossagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0

global_norm/L2Loss_6L2LossXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: 

global_norm/L2Loss_7L2Lossagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: 

global_norm/L2Loss_8L2LossXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: 
Ę
global_norm/L2Loss_9L2Loss:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
T0*
_output_shapes
: *M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1
Ă
global_norm/L2Loss_10L2Loss6gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
T0*I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
_output_shapes
: 
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
global_norm/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
T0*
_output_shapes
: 
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
clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xglobal_norm/global_norm*
T0*
_output_shapes
: 
^
clip_by_global_norm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
d
clip_by_global_norm/truediv_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @

clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
_output_shapes
: *
T0

clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
T0*
_output_shapes
: 
^
clip_by_global_norm/mul/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
_output_shapes
: *
T0
â
clip_by_global_norm/mul_1Mul2gradients/encoder/conv1d/ExpandDims_1_grad/Reshapeclip_by_global_norm/mul*
T0*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*#
_output_shapes
:
Ć
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*
T0*E
_class;
97loc:@gradients/encoder/conv1d/ExpandDims_1_grad/Reshape*#
_output_shapes
:
Ţ
clip_by_global_norm/mul_2Mul2gradients/encoder_1/initial_state_0_tiled_grad/Sumclip_by_global_norm/mul*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
:	*
T0
Â
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_0_tiled_grad/Sum*
_output_shapes
:	
Ţ
clip_by_global_norm/mul_3Mul2gradients/encoder_1/initial_state_1_tiled_grad/Sumclip_by_global_norm/mul*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum*
_output_shapes
:	*
T0
Â
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3*
T0*
_output_shapes
:	*E
_class;
97loc:@gradients/encoder_1/initial_state_1_tiled_grad/Sum
Ţ
clip_by_global_norm/mul_4Mul2gradients/encoder_1/initial_state_2_tiled_grad/Sumclip_by_global_norm/mul*
T0*
_output_shapes
:	*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum
Â
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
_output_shapes
:	*E
_class;
97loc:@gradients/encoder_1/initial_state_2_tiled_grad/Sum*
T0
Ţ
clip_by_global_norm/mul_5Mul2gradients/encoder_1/initial_state_3_tiled_grad/Sumclip_by_global_norm/mul*
T0*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum*
_output_shapes
:	
Â
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*
T0*
_output_shapes
:	*E
_class;
97loc:@gradients/encoder_1/initial_state_3_tiled_grad/Sum
˝
clip_by_global_norm/mul_6Mulagradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul* 
_output_shapes
:
*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
ň
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
*
T0
Ś
clip_by_global_norm/mul_7MulXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0*
_output_shapes	
:*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3
ä
*clip_by_global_norm/clip_by_global_norm/_6Identityclip_by_global_norm/mul_7*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:*
T0
˝
clip_by_global_norm/mul_8Mulagradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul* 
_output_shapes
:
*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
ň
*clip_by_global_norm/clip_by_global_norm/_7Identityclip_by_global_norm/mul_8*t
_classj
hfloc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
*
T0
Ś
clip_by_global_norm/mul_9MulXgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:
ä
*clip_by_global_norm/clip_by_global_norm/_8Identityclip_by_global_norm/mul_9*
T0*
_output_shapes	
:*k
_classa
_]loc:@gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3
ď
clip_by_global_norm/mul_10Mul:gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1clip_by_global_norm/mul*
T0*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
_output_shapes
:	
Ë
*clip_by_global_norm/clip_by_global_norm/_9Identityclip_by_global_norm/mul_10*
T0*M
_classC
A?loc:@gradients/output_projection/xw_plus_b/MatMul_grad/MatMul_1*
_output_shapes
:	
â
clip_by_global_norm/mul_11Mul6gradients/output_projection/xw_plus_b_grad/BiasAddGradclip_by_global_norm/mul*I
_class?
=;loc:@gradients/output_projection/xw_plus_b_grad/BiasAddGrad*
_output_shapes
:*
T0
Ă
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
Ž
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
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
 *wž?*
_class
loc:@input_embed

beta2_power
VariableV2*
shape: *
_output_shapes
: *
shared_name *
_class
loc:@input_embed*
dtype0*
	container 
Ž
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
valueB*    *
dtype0*#
_output_shapes
:
Ž
input_embed/Adam
VariableV2*
	container *
dtype0*
_class
loc:@input_embed*
shared_name *#
_output_shapes
:*
shape:
ą
input_embed/Adam/AssignAssigninput_embed/Adamzeros*
use_locking(*
validate_shape(*
T0*#
_output_shapes
:*
_class
loc:@input_embed

input_embed/Adam/readIdentityinput_embed/Adam*
T0*#
_output_shapes
:*
_class
loc:@input_embed
f
zeros_1Const*"
valueB*    *#
_output_shapes
:*
dtype0
°
input_embed/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@input_embed*#
_output_shapes
:*
shape:*
shared_name 
ˇ
input_embed/Adam_1/AssignAssigninput_embed/Adam_1zeros_1*
_class
loc:@input_embed*#
_output_shapes
:*
T0*
validate_shape(*
use_locking(

input_embed/Adam_1/readIdentityinput_embed/Adam_1*#
_output_shapes
:*
_class
loc:@input_embed*
T0
^
zeros_2Const*
valueB	*    *
dtype0*
_output_shapes
:	
ž
encoder/initial_state_0/Adam
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_0*
shared_name *
_output_shapes
:	*
shape:	
Ó
#encoder/initial_state_0/Adam/AssignAssignencoder/initial_state_0/Adamzeros_2**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
Ą
!encoder/initial_state_0/Adam/readIdentityencoder/initial_state_0/Adam*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0
^
zeros_3Const*
dtype0*
_output_shapes
:	*
valueB	*    
Ŕ
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
%encoder/initial_state_0/Adam_1/AssignAssignencoder/initial_state_0/Adam_1zeros_3*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(
Ľ
#encoder/initial_state_0/Adam_1/readIdentityencoder/initial_state_0/Adam_1**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
T0
^
zeros_4Const*
valueB	*    *
dtype0*
_output_shapes
:	
ž
encoder/initial_state_1/Adam
VariableV2*
	container *
dtype0**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	*
shape:	*
shared_name 
Ó
#encoder/initial_state_1/Adam/AssignAssignencoder/initial_state_1/Adamzeros_4**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
Ą
!encoder/initial_state_1/Adam/readIdentityencoder/initial_state_1/Adam*
T0**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	
^
zeros_5Const*
_output_shapes
:	*
dtype0*
valueB	*    
Ŕ
encoder/initial_state_1/Adam_1
VariableV2*
_output_shapes
:	*
dtype0*
shape:	*
	container **
_class 
loc:@encoder/initial_state_1*
shared_name 
×
%encoder/initial_state_1/Adam_1/AssignAssignencoder/initial_state_1/Adam_1zeros_5*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1
Ľ
#encoder/initial_state_1/Adam_1/readIdentityencoder/initial_state_1/Adam_1**
_class 
loc:@encoder/initial_state_1*
_output_shapes
:	*
T0
^
zeros_6Const*
valueB	*    *
dtype0*
_output_shapes
:	
ž
encoder/initial_state_2/Adam
VariableV2**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	*
shape:	*
dtype0*
shared_name *
	container 
Ó
#encoder/initial_state_2/Adam/AssignAssignencoder/initial_state_2/Adamzeros_6**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
Ą
!encoder/initial_state_2/Adam/readIdentityencoder/initial_state_2/Adam*
T0**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	
^
zeros_7Const*
valueB	*    *
dtype0*
_output_shapes
:	
Ŕ
encoder/initial_state_2/Adam_1
VariableV2*
shared_name **
_class 
loc:@encoder/initial_state_2*
	container *
shape:	*
dtype0*
_output_shapes
:	
×
%encoder/initial_state_2/Adam_1/AssignAssignencoder/initial_state_2/Adam_1zeros_7*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_2*
validate_shape(*
_output_shapes
:	
Ľ
#encoder/initial_state_2/Adam_1/readIdentityencoder/initial_state_2/Adam_1*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2*
T0
^
zeros_8Const*
_output_shapes
:	*
dtype0*
valueB	*    
ž
encoder/initial_state_3/Adam
VariableV2*
shape:	*
_output_shapes
:	*
shared_name **
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
Ą
!encoder/initial_state_3/Adam/readIdentityencoder/initial_state_3/Adam*
T0**
_class 
loc:@encoder/initial_state_3*
_output_shapes
:	
^
zeros_9Const*
valueB	*    *
_output_shapes
:	*
dtype0
Ŕ
encoder/initial_state_3/Adam_1
VariableV2*
shared_name **
_class 
loc:@encoder/initial_state_3*
	container *
shape:	*
dtype0*
_output_shapes
:	
×
%encoder/initial_state_3/Adam_1/AssignAssignencoder/initial_state_3/Adam_1zeros_9*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_3*
T0*
use_locking(
Ľ
#encoder/initial_state_3/Adam_1/readIdentityencoder/initial_state_3/Adam_1*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3*
T0
a
zeros_10Const* 
_output_shapes
:
*
dtype0*
valueB
*    
ř
8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam
VariableV2*
	container *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
shared_name * 
_output_shapes
:
*
shape:

Š
?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/AssignAssign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamzeros_10* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0*
use_locking(
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
valueB
*    * 
_output_shapes
:
*
dtype0
ú
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1
VariableV2*
shared_name *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
	container *
shape:
*
dtype0* 
_output_shapes
:

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
ú
?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/readIdentity:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0
W
zeros_12Const*
dtype0*
_output_shapes	
:*
valueB*    
ě
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam
VariableV2*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
	container *
shape:*
dtype0*
_output_shapes	
:
Ą
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adamzeros_12*
_output_shapes	
:*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
T0*
use_locking(
î
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
î
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1
VariableV2*
	container *
dtype0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
shape:*
shared_name 
Ľ
@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1zeros_13*
_output_shapes	
:*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
T0*
use_locking(
ň
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/readIdentity9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:
a
zeros_14Const*
valueB
*    *
dtype0* 
_output_shapes
:

ř
8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam
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
Š
?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/AssignAssign8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adamzeros_14*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
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
valueB
*    *
dtype0* 
_output_shapes
:

ú
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1
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
ú
?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/readIdentity:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
T0
W
zeros_16Const*
dtype0*
_output_shapes	
:*
valueB*    
ě
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam
VariableV2*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
	container *
shape:*
dtype0*
_output_shapes	
:
Ą
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adamzeros_16*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
î
<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/readIdentity7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0
W
zeros_17Const*
_output_shapes	
:*
dtype0*
valueB*    
î
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1
VariableV2*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
	container *
shape:*
dtype0*
_output_shapes	
:
Ľ
@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1zeros_17*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
ň
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/readIdentity9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes	
:*
T0
_
zeros_18Const*
valueB	*    *
_output_shapes
:	*
dtype0
ś
output_projection/W/Adam
VariableV2*
	container *
dtype0*&
_class
loc:@output_projection/W*
shared_name *
_output_shapes
:	*
shape:	
Č
output_projection/W/Adam/AssignAssignoutput_projection/W/Adamzeros_18*
use_locking(*
T0*&
_class
loc:@output_projection/W*
validate_shape(*
_output_shapes
:	

output_projection/W/Adam/readIdentityoutput_projection/W/Adam*
T0*&
_class
loc:@output_projection/W*
_output_shapes
:	
_
zeros_19Const*
valueB	*    *
dtype0*
_output_shapes
:	
¸
output_projection/W/Adam_1
VariableV2*&
_class
loc:@output_projection/W*
_output_shapes
:	*
shape:	*
dtype0*
shared_name *
	container 
Ě
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
zeros_20Const*
_output_shapes
:*
dtype0*
valueB*    
Ź
output_projection/b/Adam
VariableV2*
shape:*
_output_shapes
:*
shared_name *&
_class
loc:@output_projection/b*
dtype0*
	container 
Ă
output_projection/b/Adam/AssignAssignoutput_projection/b/Adamzeros_20*&
_class
loc:@output_projection/b*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

output_projection/b/Adam/readIdentityoutput_projection/b/Adam*
T0*&
_class
loc:@output_projection/b*
_output_shapes
:
U
zeros_21Const*
valueB*    *
_output_shapes
:*
dtype0
Ž
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
Ç
!output_projection/b/Adam_1/AssignAssignoutput_projection/b/Adam_1zeros_21*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(

output_projection/b/Adam_1/readIdentityoutput_projection/b/Adam_1*
T0*
_output_shapes
:*&
_class
loc:@output_projection/b
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
valueB
 *wž?*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
Ç
!Adam/update_input_embed/ApplyAdam	ApplyAdaminput_embedinput_embed/Adaminput_embed/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
_class
loc:@input_embed*#
_output_shapes
:*
T0*
use_locking( 
˙
-Adam/update_encoder/initial_state_0/ApplyAdam	ApplyAdamencoder/initial_state_0encoder/initial_state_0/Adamencoder/initial_state_0/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_locking( *
T0**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	
˙
-Adam/update_encoder/initial_state_1/ApplyAdam	ApplyAdamencoder/initial_state_1encoder/initial_state_1/Adamencoder/initial_state_1/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1*
T0*
use_locking( 
˙
-Adam/update_encoder/initial_state_2/ApplyAdam	ApplyAdamencoder/initial_state_2encoder/initial_state_2/Adamencoder/initial_state_2/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
use_locking( *
T0**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	
˙
-Adam/update_encoder/initial_state_3/ApplyAdam	ApplyAdamencoder/initial_state_3encoder/initial_state_3/Adamencoder/initial_state_3/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_4*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_3*
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
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights

HAdam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam	ApplyAdam2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_8*
use_locking( *
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
ë
)Adam/update_output_projection/W/ApplyAdam	ApplyAdamoutput_projection/Woutput_projection/W/Adamoutput_projection/W/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_9*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0*
use_locking( 
ç
)Adam/update_output_projection/b/ApplyAdam	ApplyAdamoutput_projection/boutput_projection/b/Adamoutput_projection/b/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_10*
use_locking( *
T0*
_output_shapes
:*&
_class
loc:@output_projection/b
Ř
Adam/mulMulbeta1_power/read
Adam/beta1"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam*
T0*
_class
loc:@input_embed*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@input_embed*
validate_shape(*
_output_shapes
: 
Ú

Adam/mul_1Mulbeta2_power/read
Adam/beta2"^Adam/update_input_embed/ApplyAdam.^Adam/update_encoder/initial_state_0/ApplyAdam.^Adam/update_encoder/initial_state_1/ApplyAdam.^Adam/update_encoder/initial_state_2/ApplyAdam.^Adam/update_encoder/initial_state_3/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/ApplyAdamJ^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/ApplyAdam*^Adam/update_output_projection/W/ApplyAdam*^Adam/update_output_projection/b/ApplyAdam*
_class
loc:@input_embed*
_output_shapes
: *
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

Adam/valueConst^Adam/update*
value	B :*
_class
loc:@Variable*
_output_shapes
: *
dtype0
x
Adam	AssignAddVariable
Adam/value*
_class
loc:@Variable*
_output_shapes
: *
T0*
use_locking( 
Z
train_loss/tagsConst*
valueB B
train_loss*
dtype0*
_output_shapes
: 
X

train_lossScalarSummarytrain_loss/tags	loss/Mean*
T0*
_output_shapes
: 
b
train_accuracy/tagsConst*
dtype0*
_output_shapes
: *
valueB Btrain_accuracy
h
train_accuracyScalarSummarytrain_accuracy/tagsaccuracy/accuracy*
T0*
_output_shapes
: 
_
Merge/MergeSummaryMergeSummary
train_losstrain_accuracy*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
Ń

save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:$*

valueú	B÷	$BVariableBbeta1_powerBbeta2_powerBencoder/initial_state_0Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1Binput_embedBinput_embed/AdamBinput_embed/Adam_1Boutput_projection/WBoutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/bBoutput_projection/b/AdamBoutput_projection/b/Adam_1
Ť
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
T0*
_class
loc:@save/Const*
_output_shapes
: 
l
save/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:*
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
valueBBbeta1_power*
_output_shapes
:*
dtype0
j
!save/RestoreV2_1/shape_and_slicesConst*
_output_shapes
:*
dtype0*
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
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/RestoreV2_3/tensor_namesConst*
_output_shapes
:*
dtype0*,
value#B!Bencoder/initial_state_0
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
Á
save/Assign_3Assignencoder/initial_state_0save/RestoreV2_3*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0

save/RestoreV2_4/tensor_namesConst*1
value(B&Bencoder/initial_state_0/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
Ć
save/Assign_4Assignencoder/initial_state_0/Adamsave/RestoreV2_4**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
T0*
validate_shape(*
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
B *
dtype0*
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
Č
save/Assign_5Assignencoder/initial_state_0/Adam_1save/RestoreV2_5*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_0*
validate_shape(*
_output_shapes
:	
}
save/RestoreV2_6/tensor_namesConst*,
value#B!Bencoder/initial_state_1*
_output_shapes
:*
dtype0
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
Á
save/Assign_6Assignencoder/initial_state_1save/RestoreV2_6*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	

save/RestoreV2_7/tensor_namesConst*
dtype0*
_output_shapes
:*1
value(B&Bencoder/initial_state_1/Adam
j
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ć
save/Assign_7Assignencoder/initial_state_1/Adamsave/RestoreV2_7*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1

save/RestoreV2_8/tensor_namesConst*
dtype0*
_output_shapes
:*3
value*B(Bencoder/initial_state_1/Adam_1
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
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
value#B!Bencoder/initial_state_2*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
Á
save/Assign_9Assignencoder/initial_state_2save/RestoreV2_9*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_2

save/RestoreV2_10/tensor_namesConst*1
value(B&Bencoder/initial_state_2/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
Č
save/Assign_10Assignencoder/initial_state_2/Adamsave/RestoreV2_10*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_2*
validate_shape(*
_output_shapes
:	

save/RestoreV2_11/tensor_namesConst*
dtype0*
_output_shapes
:*3
value*B(Bencoder/initial_state_2/Adam_1
k
"save/RestoreV2_11/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save/Assign_11Assignencoder/initial_state_2/Adam_1save/RestoreV2_11**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
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
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
Ă
save/Assign_12Assignencoder/initial_state_3save/RestoreV2_12*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	
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
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
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
"save/RestoreV2_14/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save/Assign_14Assignencoder/initial_state_3/Adam_1save/RestoreV2_14*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	

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

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
_output_shapes
:*
dtypes
2
ő
save/Assign_15Assign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasessave/RestoreV2_15*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases

save/RestoreV2_16/tensor_namesConst*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
ú
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
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
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
value?B=B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
dtype0*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
ü
save/Assign_18Assign3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightssave/RestoreV2_18*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
validate_shape(* 
_output_shapes
:


save/RestoreV2_19/tensor_namesConst*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_19/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:

save/Assign_19Assign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamsave/RestoreV2_19* 
_output_shapes
:
*
validate_shape(*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
T0*
use_locking(
Ą
save/RestoreV2_20/tensor_namesConst*
_output_shapes
:*
dtype0*O
valueFBDB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1
k
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:

save/Assign_20Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1save/RestoreV2_20*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
validate_shape(* 
_output_shapes
:


save/RestoreV2_21/tensor_namesConst*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes
:*
dtype0
k
"save/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ő
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
B *
dtype0*
_output_shapes
:

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
_output_shapes
:*
dtypes
2
ú
save/Assign_22Assign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adamsave/RestoreV2_22*
_output_shapes	
:*
validate_shape(*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
T0*
use_locking(
 
save/RestoreV2_23/tensor_namesConst*
dtype0*
_output_shapes
:*N
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
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
_output_shapes
:*
dtypes
2
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
"save/RestoreV2_24/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/RestoreV2_25/tensor_namesConst*
dtype0*
_output_shapes
:*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam
k
"save/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_25Assign8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adamsave/RestoreV2_25*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
validate_shape(* 
_output_shapes
:

Ą
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

save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:

save/Assign_26Assign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1save/RestoreV2_26*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
r
save/RestoreV2_27/tensor_namesConst* 
valueBBinput_embed*
dtype0*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_27Assigninput_embedsave/RestoreV2_27*#
_output_shapes
:*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
w
save/RestoreV2_28/tensor_namesConst*%
valueBBinput_embed/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_28/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
_output_shapes
:*
dtypes
2
´
save/Assign_28Assigninput_embed/Adamsave/RestoreV2_28*#
_output_shapes
:*
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
ś
save/Assign_29Assigninput_embed/Adam_1save/RestoreV2_29*
_class
loc:@input_embed*#
_output_shapes
:*
T0*
validate_shape(*
use_locking(
z
save/RestoreV2_30/tensor_namesConst*
dtype0*
_output_shapes
:*(
valueBBoutput_projection/W
k
"save/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
_output_shapes
:*
dtypes
2
ť
save/Assign_30Assignoutput_projection/Wsave/RestoreV2_30*
use_locking(*
T0*&
_class
loc:@output_projection/W*
validate_shape(*
_output_shapes
:	
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
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
Ŕ
save/Assign_31Assignoutput_projection/W/Adamsave/RestoreV2_31*&
_class
loc:@output_projection/W*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

save/RestoreV2_32/tensor_namesConst*
dtype0*
_output_shapes
:*/
value&B$Boutput_projection/W/Adam_1
k
"save/RestoreV2_32/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
_output_shapes
:*
dtypes
2
Â
save/Assign_32Assignoutput_projection/W/Adam_1save/RestoreV2_32*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*&
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
B *
dtype0*
_output_shapes
:

save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
_output_shapes
:*
dtypes
2
ś
save/Assign_33Assignoutput_projection/bsave/RestoreV2_33*
use_locking(*
T0*&
_class
loc:@output_projection/b*
validate_shape(*
_output_shapes
:

save/RestoreV2_34/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Boutput_projection/b/Adam
k
"save/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
ť
save/Assign_34Assignoutput_projection/b/Adamsave/RestoreV2_34*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(
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
˝
save/Assign_35Assignoutput_projection/b/Adam_1save/RestoreV2_35*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(
đ
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35
f
train/total_loss/tagsConst*
dtype0*
_output_shapes
: *!
valueB Btrain/total_loss
c
train/total_lossScalarSummarytrain/total_loss/tagsloss/Sum*
T0*
_output_shapes
: 
V
train/lr/tagsConst*
dtype0*
_output_shapes
: *
valueB Btrain/lr
X
train/lrScalarSummarytrain/lr/tagslearning_rate*
T0*
_output_shapes
: 
a
Merge_1/MergeSummaryMergeSummarytrain/total_losstrain/lr*
N*
_output_shapes
: 
d
test/total_loss/tagsConst*
dtype0*
_output_shapes
: * 
valueB Btest/total_loss
a
test/total_lossScalarSummarytest/total_loss/tagsloss/Sum*
_output_shapes
: *
T0
V
Merge_2/MergeSummaryMergeSummarytest/total_loss*
_output_shapes
: *
N
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Ó

save_1/SaveV2/tensor_namesConst*

valueú	B÷	$BVariableBbeta1_powerBbeta2_powerBencoder/initial_state_0Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1B3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1Binput_embedBinput_embed/AdamBinput_embed/Adam_1Boutput_projection/WBoutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/bBoutput_projection/b/AdamBoutput_projection/b/Adam_1*
dtype0*
_output_shapes
:$
­
save_1/SaveV2/shape_and_slicesConst*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:$*
dtype0

save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesVariablebeta1_powerbeta2_powerencoder/initial_state_0encoder/initial_state_0/Adamencoder/initial_state_0/Adam_1encoder/initial_state_1encoder/initial_state_1/Adamencoder/initial_state_1/Adam_1encoder/initial_state_2encoder/initial_state_2/Adamencoder/initial_state_2/Adam_1encoder/initial_state_3encoder/initial_state_3/Adamencoder/initial_state_3/Adam_12encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_13encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_12encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_13encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1input_embedinput_embed/Adaminput_embed/Adam_1output_projection/Woutput_projection/W/Adamoutput_projection/W/Adam_1output_projection/boutput_projection/b/Adamoutput_projection/b/Adam_1*2
dtypes(
&2$

save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
_output_shapes
: *
_class
loc:@save_1/Const*
T0
n
save_1/RestoreV2/tensor_namesConst*
valueBBVariable*
_output_shapes
:*
dtype0
j
!save_1/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2

save_1/AssignAssignVariablesave_1/RestoreV2*
_class
loc:@Variable*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
s
save_1/RestoreV2_1/tensor_namesConst*
_output_shapes
:*
dtype0* 
valueBBbeta1_power
l
#save_1/RestoreV2_1/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
¤
save_1/Assign_1Assignbeta1_powersave_1/RestoreV2_1*
use_locking(*
T0*
_class
loc:@input_embed*
validate_shape(*
_output_shapes
: 
s
save_1/RestoreV2_2/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save_1/RestoreV2_2	RestoreV2save_1/Constsave_1/RestoreV2_2/tensor_names#save_1/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
¤
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@input_embed

save_1/RestoreV2_3/tensor_namesConst*,
value#B!Bencoder/initial_state_0*
_output_shapes
:*
dtype0
l
#save_1/RestoreV2_3/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save_1/RestoreV2_3	RestoreV2save_1/Constsave_1/RestoreV2_3/tensor_names#save_1/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
Ĺ
save_1/Assign_3Assignencoder/initial_state_0save_1/RestoreV2_3*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_0

save_1/RestoreV2_4/tensor_namesConst*1
value(B&Bencoder/initial_state_0/Adam*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_4/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save_1/RestoreV2_4	RestoreV2save_1/Constsave_1/RestoreV2_4/tensor_names#save_1/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
Ę
save_1/Assign_4Assignencoder/initial_state_0/Adamsave_1/RestoreV2_4**
_class 
loc:@encoder/initial_state_0*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

save_1/RestoreV2_5/tensor_namesConst*
dtype0*
_output_shapes
:*3
value*B(Bencoder/initial_state_0/Adam_1
l
#save_1/RestoreV2_5/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save_1/RestoreV2_5	RestoreV2save_1/Constsave_1/RestoreV2_5/tensor_names#save_1/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
Ě
save_1/Assign_5Assignencoder/initial_state_0/Adam_1save_1/RestoreV2_5*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_0*
T0*
use_locking(

save_1/RestoreV2_6/tensor_namesConst*,
value#B!Bencoder/initial_state_1*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_6	RestoreV2save_1/Constsave_1/RestoreV2_6/tensor_names#save_1/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ĺ
save_1/Assign_6Assignencoder/initial_state_1save_1/RestoreV2_6*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	

save_1/RestoreV2_7/tensor_namesConst*
dtype0*
_output_shapes
:*1
value(B&Bencoder/initial_state_1/Adam
l
#save_1/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_7	RestoreV2save_1/Constsave_1/RestoreV2_7/tensor_names#save_1/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
Ę
save_1/Assign_7Assignencoder/initial_state_1/Adamsave_1/RestoreV2_7*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_1*
validate_shape(*
_output_shapes
:	

save_1/RestoreV2_8/tensor_namesConst*3
value*B(Bencoder/initial_state_1/Adam_1*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_8/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2_8	RestoreV2save_1/Constsave_1/RestoreV2_8/tensor_names#save_1/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ě
save_1/Assign_8Assignencoder/initial_state_1/Adam_1save_1/RestoreV2_8*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	**
_class 
loc:@encoder/initial_state_1

save_1/RestoreV2_9/tensor_namesConst*
_output_shapes
:*
dtype0*,
value#B!Bencoder/initial_state_2
l
#save_1/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_9	RestoreV2save_1/Constsave_1/RestoreV2_9/tensor_names#save_1/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
Ĺ
save_1/Assign_9Assignencoder/initial_state_2save_1/RestoreV2_9*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_2*
T0*
use_locking(

 save_1/RestoreV2_10/tensor_namesConst*1
value(B&Bencoder/initial_state_2/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_10/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_10	RestoreV2save_1/Const save_1/RestoreV2_10/tensor_names$save_1/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ě
save_1/Assign_10Assignencoder/initial_state_2/Adamsave_1/RestoreV2_10*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_2*
T0*
use_locking(

 save_1/RestoreV2_11/tensor_namesConst*3
value*B(Bencoder/initial_state_2/Adam_1*
_output_shapes
:*
dtype0
m
$save_1/RestoreV2_11/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
Ą
save_1/RestoreV2_11	RestoreV2save_1/Const save_1/RestoreV2_11/tensor_names$save_1/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
Î
save_1/Assign_11Assignencoder/initial_state_2/Adam_1save_1/RestoreV2_11**
_class 
loc:@encoder/initial_state_2*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

 save_1/RestoreV2_12/tensor_namesConst*
dtype0*
_output_shapes
:*,
value#B!Bencoder/initial_state_3
m
$save_1/RestoreV2_12/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ą
save_1/RestoreV2_12	RestoreV2save_1/Const save_1/RestoreV2_12/tensor_names$save_1/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
Ç
save_1/Assign_12Assignencoder/initial_state_3save_1/RestoreV2_12*
use_locking(*
T0**
_class 
loc:@encoder/initial_state_3*
validate_shape(*
_output_shapes
:	

 save_1/RestoreV2_13/tensor_namesConst*
dtype0*
_output_shapes
:*1
value(B&Bencoder/initial_state_3/Adam
m
$save_1/RestoreV2_13/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_13	RestoreV2save_1/Const save_1/RestoreV2_13/tensor_names$save_1/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
Ě
save_1/Assign_13Assignencoder/initial_state_3/Adamsave_1/RestoreV2_13*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_3*
T0*
use_locking(

 save_1/RestoreV2_14/tensor_namesConst*
dtype0*
_output_shapes
:*3
value*B(Bencoder/initial_state_3/Adam_1
m
$save_1/RestoreV2_14/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_14	RestoreV2save_1/Const save_1/RestoreV2_14/tensor_names$save_1/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Î
save_1/Assign_14Assignencoder/initial_state_3/Adam_1save_1/RestoreV2_14*
_output_shapes
:	*
validate_shape(**
_class 
loc:@encoder/initial_state_3*
T0*
use_locking(

 save_1/RestoreV2_15/tensor_namesConst*
dtype0*
_output_shapes
:*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
m
$save_1/RestoreV2_15/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ą
save_1/RestoreV2_15	RestoreV2save_1/Const save_1/RestoreV2_15/tensor_names$save_1/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
ů
save_1/Assign_15Assign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasessave_1/RestoreV2_15*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
validate_shape(*
_output_shapes	
:
 
 save_1/RestoreV2_16/tensor_namesConst*
dtype0*
_output_shapes
:*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam
m
$save_1/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_16	RestoreV2save_1/Const save_1/RestoreV2_16/tensor_names$save_1/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
ţ
save_1/Assign_16Assign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adamsave_1/RestoreV2_16*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
˘
 save_1/RestoreV2_17/tensor_namesConst*
dtype0*
_output_shapes
:*N
valueEBCB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1
m
$save_1/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_17	RestoreV2save_1/Const save_1/RestoreV2_17/tensor_names$save_1/RestoreV2_17/shape_and_slices*
_output_shapes
:*
dtypes
2

save_1/Assign_17Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1save_1/RestoreV2_17*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(

 save_1/RestoreV2_18/tensor_namesConst*H
value?B=B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
_output_shapes
:*
dtype0
m
$save_1/RestoreV2_18/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ą
save_1/RestoreV2_18	RestoreV2save_1/Const save_1/RestoreV2_18/tensor_names$save_1/RestoreV2_18/shape_and_slices*
_output_shapes
:*
dtypes
2

save_1/Assign_18Assign3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightssave_1/RestoreV2_18*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ą
 save_1/RestoreV2_19/tensor_namesConst*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_19/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
Ą
save_1/RestoreV2_19	RestoreV2save_1/Const save_1/RestoreV2_19/tensor_names$save_1/RestoreV2_19/shape_and_slices*
_output_shapes
:*
dtypes
2

save_1/Assign_19Assign8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adamsave_1/RestoreV2_19*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ł
 save_1/RestoreV2_20/tensor_namesConst*O
valueFBDB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_20/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Ą
save_1/RestoreV2_20	RestoreV2save_1/Const save_1/RestoreV2_20/tensor_names$save_1/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:

save_1/Assign_20Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1save_1/RestoreV2_20*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights

 save_1/RestoreV2_21/tensor_namesConst*G
value>B<B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_21/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
Ą
save_1/RestoreV2_21	RestoreV2save_1/Const save_1/RestoreV2_21/tensor_names$save_1/RestoreV2_21/shape_and_slices*
_output_shapes
:*
dtypes
2
ů
save_1/Assign_21Assign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasessave_1/RestoreV2_21*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
 
 save_1/RestoreV2_22/tensor_namesConst*L
valueCBAB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_22	RestoreV2save_1/Const save_1/RestoreV2_22/tensor_names$save_1/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ţ
save_1/Assign_22Assign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adamsave_1/RestoreV2_22*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases
˘
 save_1/RestoreV2_23/tensor_namesConst*
_output_shapes
:*
dtype0*N
valueEBCB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1
m
$save_1/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_23	RestoreV2save_1/Const save_1/RestoreV2_23/tensor_names$save_1/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:

save_1/Assign_23Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1save_1/RestoreV2_23*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
validate_shape(*
_output_shapes	
:

 save_1/RestoreV2_24/tensor_namesConst*
dtype0*
_output_shapes
:*H
value?B=B3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
m
$save_1/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_24	RestoreV2save_1/Const save_1/RestoreV2_24/tensor_names$save_1/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:

save_1/Assign_24Assign3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightssave_1/RestoreV2_24*
use_locking(*
T0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
validate_shape(* 
_output_shapes
:

Ą
 save_1/RestoreV2_25/tensor_namesConst*
dtype0*
_output_shapes
:*M
valueDBBB8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam
m
$save_1/RestoreV2_25/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
Ą
save_1/RestoreV2_25	RestoreV2save_1/Const save_1/RestoreV2_25/tensor_names$save_1/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:

save_1/Assign_25Assign8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adamsave_1/RestoreV2_25*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ł
 save_1/RestoreV2_26/tensor_namesConst*
_output_shapes
:*
dtype0*O
valueFBDB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1
m
$save_1/RestoreV2_26/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
Ą
save_1/RestoreV2_26	RestoreV2save_1/Const save_1/RestoreV2_26/tensor_names$save_1/RestoreV2_26/shape_and_slices*
_output_shapes
:*
dtypes
2

save_1/Assign_26Assign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1save_1/RestoreV2_26*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
t
 save_1/RestoreV2_27/tensor_namesConst*
_output_shapes
:*
dtype0* 
valueBBinput_embed
m
$save_1/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_27	RestoreV2save_1/Const save_1/RestoreV2_27/tensor_names$save_1/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ł
save_1/Assign_27Assigninput_embedsave_1/RestoreV2_27*
use_locking(*
validate_shape(*
T0*#
_output_shapes
:*
_class
loc:@input_embed
y
 save_1/RestoreV2_28/tensor_namesConst*
_output_shapes
:*
dtype0*%
valueBBinput_embed/Adam
m
$save_1/RestoreV2_28/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ą
save_1/RestoreV2_28	RestoreV2save_1/Const save_1/RestoreV2_28/tensor_names$save_1/RestoreV2_28/shape_and_slices*
_output_shapes
:*
dtypes
2
¸
save_1/Assign_28Assigninput_embed/Adamsave_1/RestoreV2_28*#
_output_shapes
:*
validate_shape(*
_class
loc:@input_embed*
T0*
use_locking(
{
 save_1/RestoreV2_29/tensor_namesConst*
dtype0*
_output_shapes
:*'
valueBBinput_embed/Adam_1
m
$save_1/RestoreV2_29/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
Ą
save_1/RestoreV2_29	RestoreV2save_1/Const save_1/RestoreV2_29/tensor_names$save_1/RestoreV2_29/shape_and_slices*
_output_shapes
:*
dtypes
2
ş
save_1/Assign_29Assigninput_embed/Adam_1save_1/RestoreV2_29*
use_locking(*
T0*
_class
loc:@input_embed*
validate_shape(*#
_output_shapes
:
|
 save_1/RestoreV2_30/tensor_namesConst*(
valueBBoutput_projection/W*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_30/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
Ą
save_1/RestoreV2_30	RestoreV2save_1/Const save_1/RestoreV2_30/tensor_names$save_1/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
ż
save_1/Assign_30Assignoutput_projection/Wsave_1/RestoreV2_30*
_output_shapes
:	*
validate_shape(*&
_class
loc:@output_projection/W*
T0*
use_locking(

 save_1/RestoreV2_31/tensor_namesConst*
_output_shapes
:*
dtype0*-
value$B"Boutput_projection/W/Adam
m
$save_1/RestoreV2_31/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
Ą
save_1/RestoreV2_31	RestoreV2save_1/Const save_1/RestoreV2_31/tensor_names$save_1/RestoreV2_31/shape_and_slices*
_output_shapes
:*
dtypes
2
Ä
save_1/Assign_31Assignoutput_projection/W/Adamsave_1/RestoreV2_31*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*&
_class
loc:@output_projection/W

 save_1/RestoreV2_32/tensor_namesConst*/
value&B$Boutput_projection/W/Adam_1*
_output_shapes
:*
dtype0
m
$save_1/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_32	RestoreV2save_1/Const save_1/RestoreV2_32/tensor_names$save_1/RestoreV2_32/shape_and_slices*
_output_shapes
:*
dtypes
2
Ć
save_1/Assign_32Assignoutput_projection/W/Adam_1save_1/RestoreV2_32*
use_locking(*
T0*&
_class
loc:@output_projection/W*
validate_shape(*
_output_shapes
:	
|
 save_1/RestoreV2_33/tensor_namesConst*
dtype0*
_output_shapes
:*(
valueBBoutput_projection/b
m
$save_1/RestoreV2_33/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ą
save_1/RestoreV2_33	RestoreV2save_1/Const save_1/RestoreV2_33/tensor_names$save_1/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save_1/Assign_33Assignoutput_projection/bsave_1/RestoreV2_33*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(

 save_1/RestoreV2_34/tensor_namesConst*
_output_shapes
:*
dtype0*-
value$B"Boutput_projection/b/Adam
m
$save_1/RestoreV2_34/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
Ą
save_1/RestoreV2_34	RestoreV2save_1/Const save_1/RestoreV2_34/tensor_names$save_1/RestoreV2_34/shape_and_slices*
_output_shapes
:*
dtypes
2
ż
save_1/Assign_34Assignoutput_projection/b/Adamsave_1/RestoreV2_34*
_output_shapes
:*
validate_shape(*&
_class
loc:@output_projection/b*
T0*
use_locking(

 save_1/RestoreV2_35/tensor_namesConst*
dtype0*
_output_shapes
:*/
value&B$Boutput_projection/b/Adam_1
m
$save_1/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ą
save_1/RestoreV2_35	RestoreV2save_1/Const save_1/RestoreV2_35/tensor_names$save_1/RestoreV2_35/shape_and_slices*
_output_shapes
:*
dtypes
2
Á
save_1/Assign_35Assignoutput_projection/b/Adam_1save_1/RestoreV2_35*
use_locking(*
T0*&
_class
loc:@output_projection/b*
validate_shape(*
_output_shapes
:
ş
save_1/restore_allNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35

4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedVariable*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
Ą
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedinput_embed*
_output_shapes
: *
dtype0*
_class
loc:@input_embed
š
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedencoder/initial_state_0*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_0
š
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedencoder/initial_state_1**
_class 
loc:@encoder/initial_state_1*
dtype0*
_output_shapes
: 
š
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedencoder/initial_state_2**
_class 
loc:@encoder/initial_state_2*
_output_shapes
: *
dtype0
š
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedencoder/initial_state_3*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_3
ń
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights
ď
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
dtype0*
_output_shapes
: *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases
ń
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
dtype0*
_output_shapes
: *F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
ď
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitialized2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes
: *
dtype0
˛
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedoutput_projection/W*&
_class
loc:@output_projection/W*
_output_shapes
: *
dtype0
˛
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedoutput_projection/b*&
_class
loc:@output_projection/b*
dtype0*
_output_shapes
: 
˘
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedbeta1_power*
_class
loc:@input_embed*
_output_shapes
: *
dtype0
˘
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedbeta2_power*
_class
loc:@input_embed*
_output_shapes
: *
dtype0
§
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedinput_embed/Adam*
_class
loc:@input_embed*
_output_shapes
: *
dtype0
Š
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedinput_embed/Adam_1*
_class
loc:@input_embed*
_output_shapes
: *
dtype0
ż
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedencoder/initial_state_0/Adam**
_class 
loc:@encoder/initial_state_0*
_output_shapes
: *
dtype0
Á
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitializedencoder/initial_state_0/Adam_1**
_class 
loc:@encoder/initial_state_0*
_output_shapes
: *
dtype0
ż
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitializedencoder/initial_state_1/Adam**
_class 
loc:@encoder/initial_state_1*
_output_shapes
: *
dtype0
Á
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitializedencoder/initial_state_1/Adam_1**
_class 
loc:@encoder/initial_state_1*
_output_shapes
: *
dtype0
ż
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitializedencoder/initial_state_2/Adam**
_class 
loc:@encoder/initial_state_2*
_output_shapes
: *
dtype0
Á
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitializedencoder/initial_state_2/Adam_1*
dtype0*
_output_shapes
: **
_class 
loc:@encoder/initial_state_2
ż
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitializedencoder/initial_state_3/Adam**
_class 
loc:@encoder/initial_state_3*
dtype0*
_output_shapes
: 
Á
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitializedencoder/initial_state_3/Adam_1*
_output_shapes
: *
dtype0**
_class 
loc:@encoder/initial_state_3
÷
7report_uninitialized_variables/IsVariableInitialized_24IsVariableInitialized8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
dtype0*
_output_shapes
: 
ů
7report_uninitialized_variables/IsVariableInitialized_25IsVariableInitialized:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights*
dtype0*
_output_shapes
: 
ő
7report_uninitialized_variables/IsVariableInitialized_26IsVariableInitialized7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
_output_shapes
: *
dtype0
÷
7report_uninitialized_variables/IsVariableInitialized_27IsVariableInitialized9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases*
dtype0*
_output_shapes
: 
÷
7report_uninitialized_variables/IsVariableInitialized_28IsVariableInitialized8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam*
_output_shapes
: *
dtype0*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights
ů
7report_uninitialized_variables/IsVariableInitialized_29IsVariableInitialized:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1*F
_class<
:8loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights*
_output_shapes
: *
dtype0
ő
7report_uninitialized_variables/IsVariableInitialized_30IsVariableInitialized7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes
: *
dtype0
÷
7report_uninitialized_variables/IsVariableInitialized_31IsVariableInitialized9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases*
_output_shapes
: *
dtype0
ˇ
7report_uninitialized_variables/IsVariableInitialized_32IsVariableInitializedoutput_projection/W/Adam*&
_class
loc:@output_projection/W*
_output_shapes
: *
dtype0
š
7report_uninitialized_variables/IsVariableInitialized_33IsVariableInitializedoutput_projection/W/Adam_1*&
_class
loc:@output_projection/W*
_output_shapes
: *
dtype0
ˇ
7report_uninitialized_variables/IsVariableInitialized_34IsVariableInitializedoutput_projection/b/Adam*
_output_shapes
: *
dtype0*&
_class
loc:@output_projection/b
š
7report_uninitialized_variables/IsVariableInitialized_35IsVariableInitializedoutput_projection/b/Adam_1*&
_class
loc:@output_projection/b*
dtype0*
_output_shapes
: 
Ţ
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
Ý

$report_uninitialized_variables/ConstConst*
_output_shapes
:$*
dtype0*

valueú	B÷	$BVariableBinput_embedBencoder/initial_state_0Bencoder/initial_state_1Bencoder/initial_state_2Bencoder/initial_state_3B3encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weightsB2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasesB3encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weightsB2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasesBoutput_projection/WBoutput_projection/bBbeta1_powerBbeta2_powerBinput_embed/AdamBinput_embed/Adam_1Bencoder/initial_state_0/AdamBencoder/initial_state_0/Adam_1Bencoder/initial_state_1/AdamBencoder/initial_state_1/Adam_1Bencoder/initial_state_2/AdamBencoder/initial_state_2/Adam_1Bencoder/initial_state_3/AdamBencoder/initial_state_3/Adam_1B8encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1B7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1B8encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/AdamB:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1B7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1Boutput_projection/W/AdamBoutput_projection/W/Adam_1Boutput_projection/b/AdamBoutput_projection/b/Adam_1
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:$*
_output_shapes
:*
dtype0

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ů
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
ellipsis_mask *

begin_mask*
_output_shapes
:*
end_mask *
T0*
Index0*
shrink_axis_mask *
new_axis_mask 

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
ő
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

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
_output_shapes
:*
dtype0

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
á
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
T0*
Index0*
new_axis_mask *
_output_shapes
: *
shrink_axis_mask *
ellipsis_mask *

begin_mask *
end_mask
Ż
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
_output_shapes
:*
N*

axis *
T0
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ť
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
Ë
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
T0*
Tshape0*
_output_shapes
:$

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
Ű
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
T0
*
_output_shapes
:$*
Tshape0

1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	*
squeeze_dims


2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

initNoOp^Variable/Assign^input_embed/Assign^encoder/initial_state_0/Assign^encoder/initial_state_1/Assign^encoder/initial_state_2/Assign^encoder/initial_state_3/Assign;^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Assign:^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Assign;^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Assign:^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Assign^output_projection/W/Assign^output_projection/b/Assign^beta1_power/Assign^beta2_power/Assign^input_embed/Adam/Assign^input_embed/Adam_1/Assign$^encoder/initial_state_0/Adam/Assign&^encoder/initial_state_0/Adam_1/Assign$^encoder/initial_state_1/Adam/Assign&^encoder/initial_state_1/Adam_1/Assign$^encoder/initial_state_2/Adam/Assign&^encoder/initial_state_2/Adam_1/Assign$^encoder/initial_state_3/Adam/Assign&^encoder/initial_state_3/Adam_1/Assign@^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/AssignB^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/Assign?^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/AssignA^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/Assign@^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/AssignB^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/Assign?^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/AssignA^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/Assign ^output_projection/W/Adam/Assign"^output_projection/W/Adam_1/Assign ^output_projection/b/Adam/Assign"^output_projection/b/Adam_1/Assign

init_1NoOp

init_all_tablesNoOp
-

group_depsNoOp^init_1^init_all_tables"J
save_1/Const:0save_1/control_dependency:0save_1/restore_all 5 @F8"{
	summariesn
l
grad_norms/grad_norms:0
train_loss:0
train_accuracy:0
train/total_loss:0

train/lr:0
test/total_loss:0"Ź

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
Ż
5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights:0:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read:0
Ź
4encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases:09encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read:0
Ż
5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights:0:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Assign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read:0
Ź
4encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases:09encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/read:0
O
output_projection/W:0output_projection/W/Assignoutput_projection/W/read:0
O
output_projection/b:0output_projection/b/Assignoutput_projection/b/read:0"
local_init_op


group_deps"
init_op

init"
train_op

Adam"
while_context
˙
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
>gradients/encoder_1/rnn/while/Select_1_grad/zeros_like/f_acc:0Agradients/encoder_1/rnn/while/Select_1_grad/zeros_like/RefEnter:0Ď
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnter:0ł
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/f_acc:0Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul_1/RefEnter:0Ë
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc:0egradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter:0
>gradients/encoder_1/rnn/while/Select_2_grad/zeros_like/f_acc:0Agradients/encoder_1/rnn/while/Select_2_grad/zeros_like/RefEnter:0Ď
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/RefEnter:0Ż
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul/RefEnter:0{
:gradients/encoder_1/rnn/while/Select_2_grad/Select/f_acc:0=gradients/encoder_1/rnn/while/Select_2_grad/Select/RefEnter:0Ď
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc:0ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnter:0
>gradients/encoder_1/rnn/while/Select_3_grad/zeros_like/f_acc:0Agradients/encoder_1/rnn/while/Select_3_grad/zeros_like/RefEnter:0
>gradients/encoder_1/rnn/while/Select_4_grad/zeros_like/f_acc:0Agradients/encoder_1/rnn/while/Select_4_grad/zeros_like/RefEnter:0Ż
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul/RefEnter:0Ż
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul/RefEnter:0Ż
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_grad/mul_1/RefEnter:0Ď
dgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/f_acc:0ggradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul_grad/MatMul_1/RefEnter:0
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read:0Lencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/MatMul/Enter:0ľ
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/f_acc:0Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/RefEnter:0ł
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/f_acc:0Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/mul_1/RefEnter:0ł
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/f_acc:0Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/mul_1/RefEnter:0
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read:0Cencoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter:0Ç
`gradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc:0cgradients/encoder_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/RefEnter:0Ż
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_grad/mul_1/RefEnter:0Ż
Tgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/f_acc:0Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/mul/RefEnter:0
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/read:0Cencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter:0N
encoder_1/rnn/TensorArray_1:0-encoder_1/rnn/while/TensorArrayReadV3/Enter:0{
:gradients/encoder_1/rnn/while/Select_3_grad/Select/f_acc:0=gradients/encoder_1/rnn/while/Select_3_grad/Select/RefEnter:0Ë
bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/f_acc:0egradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/ShapeN/RefEnter:0ł
Vgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/f_acc:0Ygradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/mul_1/RefEnter:0Ĺ
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/f_acc:0bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/concat_grad/mod/RefEnter:0G
encoder_1/rnn/CheckSeqLen:0(encoder_1/rnn/while/GreaterEqual/Enter:0C
encoder_1/rnn/strided_slice_2:0 encoder_1/rnn/while/Less/Enter:0
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read:0Lencoder_1/rnn/while/multi_rnn_cell/cell_1/lstm_cell/lstm_cell/MatMul/Enter:0}
Jencoder_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0/encoder_1/rnn/while/TensorArrayReadV3/Enter_1:0^
encoder_1/rnn/TensorArray:0?encoder_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Ĺ
_gradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/f_acc:0bgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/lstm_cell/concat_grad/mod/RefEnter:0{
:gradients/encoder_1/rnn/while/Select_4_grad/Select/f_acc:0=gradients/encoder_1/rnn/while/Select_4_grad/Select/RefEnter:0{
:gradients/encoder_1/rnn/while/Select_1_grad/Select/f_acc:0=gradients/encoder_1/rnn/while/Select_1_grad/Select/RefEnter:0;
encoder_1/rnn/zeros:0"encoder_1/rnn/while/Select/Enter:0ľ
Wgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/f_acc:0Zgradients/encoder_1/rnn/while/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/RefEnter:0"ň"
	variablesä"á"
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
Ż
5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights:0:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Assign:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/read:0
Ź
4encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases:09encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/read:0
Ż
5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights:0:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Assign:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/read:0
Ź
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
ž
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam:0?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/Assign?encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam/read:0
Ä
<encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1:0Aencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/AssignAencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/weights/Adam_1/read:0
ť
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam:0>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/Assign>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam/read:0
Á
;encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1:0@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/Assign@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biases/Adam_1/read:0
ž
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam:0?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/Assign?encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam/read:0
Ä
<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1:0Aencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/AssignAencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/weights/Adam_1/read:0
ť
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam:0>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/Assign>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam/read:0
Á
;encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1:0@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/Assign@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biases/Adam_1/read:0
^
output_projection/W/Adam:0output_projection/W/Adam/Assignoutput_projection/W/Adam/read:0
d
output_projection/W/Adam_1:0!output_projection/W/Adam_1/Assign!output_projection/W/Adam_1/read:0
^
output_projection/b/Adam:0output_projection/b/Adam/Assignoutput_projection/b/Adam/read:0
d
output_projection/b/Adam_1:0!output_projection/b/Adam_1/Assign!output_projection/b/Adam_1/read:0"D
ready_op8
6
4report_uninitialized_variables/boolean_mask/Gather:0ş<=	       <7¸4	<ÚWČľ?ÖA:AŽô        )íŠP	XČľ?ÖA*

Variable/sec    BžŔS       }N	ÔGqČľ?ÖA:HDhome/msheehan/s2l_lstm/runLSTM.py/s2l_2017-03-23_15-28-41/model.ckptb+á