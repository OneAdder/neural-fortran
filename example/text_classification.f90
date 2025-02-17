program simple
  !! Requires:
  !! https://github.com/LKedward/fhash/tree/master
  !! https://github.com/14NGiestas/fortran-tokenizer
  use nf, only: input, network, sgd, linear2d, mse, flatten, dense, sigmoid, relu
  use fortran_tokenizer, only: tokenizer_t, token_list, token_t
  use fhash, only: fhash_tbl_t, key=>fhash_key
  implicit none
  type(network) :: net
  type(mse) :: mean_se = mse()
  integer, parameter :: num_iterations = 20
  integer :: i, j, file_len
  type(token_t) :: token
  integer :: token_i
  type(fhash_tbl_t) :: vocab
  type(token_list), allocatable :: tokenized(:)
  type(character(len=2048)), allocatable :: texts(:)
  real, allocatable :: x(:, :)
  real, allocatable :: y(:)

  type :: imdb_entry
    character(len=2048) :: review
    character(len=8) :: target
  end type imdb_entry

  type(imdb_entry), allocatable :: imdb_dataset(:)

  if (.not. file_exists('IMDB-Dataset.csv')) then
    call download_imdb_dataset()
  end if

  file_len = get_file_length('IMDB-Dataset.csv')
  allocate(imdb_dataset(file_len))
  call load_imdb_csv('IMDB-Dataset.csv', file_len, imdb_dataset)

  texts = [imdb_dataset % review]
  allocate(tokenized(size(texts)))
  call generate_vocab(vocab, tokenized, texts)

  allocate(x(size(tokenized), 2048))
  do i = 2, size(tokenized)
    do j = 1, 2048
      token = tokenized(i) % get(j)
      if (len(token % string) == 0) then
        exit
      end if
      token_i = 0
      call vocab % get(key(token % string), token_i)
      x(i-1, j) = token_i
    end do
  end do

  allocate(y(size(imdb_dataset % target)))
  do i = 2, size(imdb_dataset % target)
    if (imdb_dataset(i) % target == 'positive') then
      y(i-1) = 1.
    else
      y(i-1) = 0.
    end if
  end do

  print *, shape(x)
  print *, shape(y)

  net = network([ &
    input(10, 2048), &
    flatten(), &
    dense(10, activation=sigmoid()), &
    dense(10, activation=relu()) &
  ])

  call net % print_info()
  do i = 0, num_iterations
    call net % forward(x)
    call net % backward(y, mean_se)
    call net % update(optimizer=sgd(learning_rate=0.05))
    print *, mean_se % eval(y, net % predict(x))
  end do
  print *, net % predict(x)
  print *, y

contains
  subroutine generate_vocab(vocab, tokenized, texts)
    type(fhash_tbl_t), intent(inout) :: vocab
    type(token_list), intent(inout) :: tokenized(:)
    type(character(len=2048)), intent(in) :: texts(:)
    type(tokenizer_t) :: tokenizer
    type(token_list) :: tokens
    type(token_t) :: token
    integer :: i, j

    tokenizer = tokenizer_t()

    do i = 2, size(texts)
      tokens = tokenizer % tokenize(trim(texts(i)))
      tokenized(i) = tokens
      do j = 1, 2048
        token = tokens % get(j)
        if (len(token % string) == 0) then
          exit
        end if
        call vocab % set(key(token % string), value=i * j)
      end do
    end do
  end subroutine generate_vocab

  subroutine download_imdb_dataset()
    integer :: cmdstat, exitstat
    character(128) :: cmdmsg

    call execute_command_line(&
        'curl -LO https://github.com/Ankit152/IMDB-sentiment-analysis/raw/refs/heads/master/IMDB-Dataset.csv',&
        wait=.true., exitstat=exitstat, cmdstat=cmdstat, cmdmsg=cmdmsg&
    )
  end subroutine download_imdb_dataset

  function file_exists(filename) result(res)
    !! Check if file exists
    character(*) :: filename
    logical :: res
    inquire(file=trim(filename), exist=res)
  end function file_exists

  subroutine load_imdb_csv(filename, file_len, dataset)
    !! Load Dataset of IMDB reviews which are positive or negative
    implicit none
    character(len=*), intent(in) :: filename
    integer, intent(in) :: file_len
    type(imdb_entry), intent(inout) :: dataset(:)
    integer :: io
    integer :: i
    character(len=2048) :: line
    open(1, file=filename)
      rewind(1)
      do i = 1, file_len
        read(1, '(A)', iostat=io) line
        if (io /= 0) exit
        read(line, *, iostat=io) dataset(i)
        if (io /= 0) exit
      end do
    close(1)
  end subroutine load_imdb_csv

  function get_file_length(filename) result(n_lines)
    !! Read file and determine its length
    implicit none
    character(len=*), intent(in) :: filename
    integer :: n_lines
    integer :: io
    n_lines = 0
    open(1, file=filename)
      do
        read(1, *, iostat=io)
        if (io/=0) exit
        n_lines = n_lines + 1
      end do
    close(1)
  end function get_file_length
end program simple
