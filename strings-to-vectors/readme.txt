# Objectives

The learning objectives of this assignment are to:
1. practice Python programming skills and use of numpy arrays
2. get familiar with submitting assignments on GitHub Classroom

# Setup

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.6 or higher)](https://www.python.org/downloads/)
* [numpy (version 1.11 or higher)](http://www.numpy.org/)
* [pytest](https://docs.pytest.org/)

If you have not used Git, Python, or Numpy before, this would be a good time to
go through some tutorials:

* [git tutorial](https://try.github.io/)
* [Python tutorial](https://docs.python.org/3/tutorial/)
* [numpy tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)

You can find many other tutorials for these tools online.

# Code

You should first clone the repository that GitHub Classroom created for you:
```
git clone git@github.com:UA-ISTA-457-SP18/strings-to-vectors-<your-username>.git
```
where `<your-username>` should be replaced with your GitHub username.
Once you have successfully cloned the repository, you can begin working on the
assignment.

You will implement an `Index` that associates objects with integer indexes.
This is a very common setup step in training neural networks, which require that
everything be expressed as numbers, not objects.

A template for the `Index` class has been provided to you in the file `nn.py`.
In the template, each method has only a documentation string, with no code in
the body of the method yet.
For example, the `objects_to_indexes` method looks like:
```python
def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
    """
    Returns a vector of the indexes associated with the input objects.

    For objects not in the vocabulary, use `start-1` as the index.

    :param object_seq: A sequence of objects
    :return: A 1-dimensional array of the object indexes.
    """
```
You will write the body of this method, manipulating the `object_seq` argument,
and any state you may have stored on the `self` object, to return the vector as
described in the documentation string.

You will need to write the bodies of 9 methods for this assignment.

# Test

Tests have been provided for you in the `test_nn.py` file.
The tests show how each of the methods of `Index` is expected to be used.
For example, the `test_indexes` test method looks like:

```python
def test_indexes():
    index = nn.Index(["four", "three", "two", "one"])
    objects = ["one", "four", "four"]
    indexes = np.array([3, 0, 0])
    assert_array_equal(index.objects_to_indexes(objects), indexes)
    assert index.indexes_to_objects(indexes) == objects
```
This tests that your code correctly associates indexes with an input vocabulary
``"four", "three", "two", "one"``, and that it can convert back and forth
between objects and indexes.

To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
platform darwin -- Python 3.5.2, pytest-3.0.5, py-1.4.32, pluggy-0.4.0
rootdir: /Users/bethard/Code/ista457/strings-to-vectors, inifile: 
collected 9 items 

test_nn.py FFFFFFFFF

=================================== FAILURES ===================================
_________________________________ test_indexes _________________________________

    def test_indexes():
        index = nn.Index(["four", "three", "two", "one"])
        objects = ["one", "four", "four"]
        indexes = np.array([3, 0, 0])
>       assert_array_equal(index.objects_to_indexes(objects), indexes)
...
E       AssertionError: 
E       Arrays are not equal
E       
E       (mismatch 100.0%)
E        x: array(None, dtype=object)
E        y: array([3, 0, 0])
...
=========================== 9 failed in 0.65 seconds ===========================

```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods in `Index`.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
platform darwin -- Python 3.5.2, pytest-3.0.5, py-1.4.32, pluggy-0.4.0
rootdir: /Users/bethard/Code/ista457/strings-to-vectors-bethard, inifile: 
collected 9 items 

test_nn.py .........

=========================== 9 passed in 0.23 seconds ===========================
```

# Submission

As you are working on the code, you should regularly `git commit` to save your
current changes locally.
You should also regularly `git push` to push all saved changes to the remote
repository on GitHub.

To submit your assignment, simply make sure that you have `git push`ed all of
your changes.
Your instructor will then be able to see, run, and test your code.

You should make a habit of checking the GitHub page for your repository to make
sure your changes have been correctly pushed there.
You may also want to check the "commits" page of your repository on GitHub:
there should be a green check mark beside your last commit, showing that your
code passes all of the given tests.
If there is a red X instead, your code is still failing some of the tests.

# Grading

Assignments will be graded primarily on their ability to pass the tests that
have been provided to you.
Assignments that pass all tests will receive at least 80% of the possible
points.
To get the remaining 20% of the points, make sure that your code is using
appropriate data structures, code duplication is minimized, variables have
meaningful names, complex pieces of code are well documented, etc.
