module nf_multihead_attention_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_activation, only: softmax
  use nf_base_layer, only: base_layer
  use nf_linear2d_layer, only: linear2d_layer

  implicit none

  private
  public :: multihead_attention_layer

  type, extends(base_layer) :: multihead_attention_layer

    !! Concrete implementation of a multihead attention layer type
    integer :: sequence_length, model_dimension, n_heads, head_size

    type(linear2d_layer) :: query_layer
    type(linear2d_layer) :: key_layer
    type(linear2d_layer) :: value_layer
    type(linear2d_layer) :: output_layer

    type(softmax) :: softmax_func

    real, allocatable :: attention_matrix(:, :, :)
    real, allocatable :: sdpa(:, :, :)
    real, allocatable :: output(:, :)

    real :: scaling_factor

    real, allocatable :: q_input(:, :)
    real, allocatable :: k_input(:, :)
    real, allocatable :: v_input(:, :)
    real, allocatable :: o_input(:, :)
  contains

    procedure :: common_backward
    procedure :: common_forward
    procedure :: get_num_params
    procedure :: get_params
    procedure :: get_gradients
    procedure :: set_params
    procedure :: init_base
    procedure :: init => init_base ! in case general MHA needs to be used

    ! FIXME: those should be private but accessible by tests
    procedure :: split_heads
    procedure :: create_attention_matrix
    procedure :: normalize_attention_matrix
    procedure :: scaled_dot_product_attention
    procedure :: combine_heads
  end type multihead_attention_layer

  interface multihead_attention_layer
    module function multihead_attention_layer_cons(sequence_length, model_dimension, n_heads) result(res)
      !! This function returns the `multihead_attention_layer` instance.
      integer, intent(in) :: sequence_length, model_dimension, n_heads
      type(multihead_attention_layer) :: res
    end function multihead_attention_layer_cons
  end interface multihead_attention_layer

  interface

    module subroutine common_backward(self, input, gradient)
      !! General backprop for MultiHead Attention mechanism
      !! Might be used for both Self and Cross Attention
      !! Self Attention: sum output gradients
      !! Cross Attention: use them separately
      class(multihead_attention_layer), intent(in out) :: self
      real, intent(in) :: input(:, :)
      real, intent(in) :: gradient(:, :)
    end subroutine common_backward

    module subroutine common_forward(self, query, key, value)
      !! General forward propagation for MultiHead Attention Mechanism
      !! Might be used for both Self and Cross Attention
      !! Self Attention: pass the same value thrice
      !! Cross Attention: pass three values for your query, key and value
      class(multihead_attention_layer), intent(in out) :: self
      real, intent(in) :: query(:, :), key(:, :), value(:, :)
    end subroutine common_forward

    module subroutine init(self, input_shape)
      !! Initialize the layer data structures.
      !!
      !! This is a deferred procedure from the `base_layer` abstract type.
      class(multihead_attention_layer), intent(in out) :: self
      integer, intent(in) :: input_shape(:)
    end subroutine init

    module function split_heads(self, input) result(output)
      !! Split inputs into heads
      !!
      !! Example with two heads:
      !! input (3, 4)
      !! output (3, 2, 2)
      class(multihead_attention_layer) :: self
      real :: input(:, :)
      real :: output(self % sequence_length, self % head_size, self % n_heads)
    end function split_heads

    module subroutine create_attention_matrix(self, query, key)
      !! Create attention matrix for query and key
      !! Output dimensions: sequence_length, sequence_length, n_heads
      class(multihead_attention_layer) :: self
      real :: query(:, :, :)
      real :: key(:, :, :)
      integer :: head
    end subroutine create_attention_matrix

    module subroutine normalize_attention_matrix(self, attention_mask)
      !! Create attention matrix for query and key
      !! Output dims: sequence_length, sequence_length, n_heads
      class(multihead_attention_layer) :: self
      !! (sequence_length, sequence_length, n_heads)
      real, optional :: attention_mask(:, :, :)
      !! (sequence_length, sequence_length, n_heads)
      real, allocatable :: output(:, :, :)
      integer :: head, seq
    end subroutine normalize_attention_matrix

    module subroutine scaled_dot_product_attention(self, value)
      !! Create scaled dot product attention
      !! Output dims: sequence_length, head_size, n_heads
      class(multihead_attention_layer) :: self
      real :: value(:, :, :)
      integer :: head
    end subroutine scaled_dot_product_attention

    module function combine_heads(self, input) result(output)
      class(multihead_attention_layer) :: self
      real :: input(:, :, :)
      !! (sequence_length, head_size, n_heads)
      real :: output(self % sequence_length, self % model_dimension)
      integer :: seq
    end function combine_heads

    module function get_num_params(self) result(num_params)
      class(multihead_attention_layer) :: self
      integer :: num_params
    end function get_num_params

    module function get_params(self) result(params)
      class(multihead_attention_layer), intent(in), target :: self
      real, allocatable :: params(:)
    end function get_params

    module function get_gradients(self) result(gradients)
      class(multihead_attention_layer), intent(in), target :: self
      real, allocatable :: gradients(:)
    end function get_gradients

    module subroutine set_params(self, params)
      class(multihead_attention_layer), intent(in out) :: self
      real, intent(in), target :: params(:)
    end subroutine set_params

    module subroutine init_base(self, input_shape)
      class(multihead_attention_layer), intent(in out) :: self
      integer, intent(in) :: input_shape(:)
    end subroutine init_base
  end interface
end module nf_multihead_attention_layer
