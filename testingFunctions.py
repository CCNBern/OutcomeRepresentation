import numpy as np
import os


def testingIsEmpty(arrayLike) -> None:

    assert len(arrayLike) != 0, "The array/list is empty!!!"


def testingIsAllZero(arrayLike) -> None:
    assert np.all((np.asarray(arrayLike) == 0)) == False, "The array/list is empty!!!"

def testingType(var, type_req) -> None:
    assert type(var) is type_req, "The data structure is not " + str(type_req) + "!!!"

def testingIsAllSame(arrayLike) -> None:
    arrayLike = np.asarray(arrayLike)
    assert arrayLike.shape[0]>1 and len(np.unique(arrayLike))>1, "All values are the same!!"  #+ str(arrayLike[0])

def testingIsNone_any(arrayLike):
    arrayLike = np.asarray(arrayLike)
    assert arrayLike.any() != None, "Error: NONE value detected!!!"

def testingIsNone_all(arrayLike) -> None:
    arrayLike = np.asarray(arrayLike)
    assert arrayLike.all() != None, "Error: All values are NONE!!!"


def testingDataShape(myArray: np.array, trueSize: int) -> None:
    assert len(myArray.shape) == trueSize, "Shape is not correct!!"

def testFolder(foldername):
    assert os.path.isdir(foldername), f"Folder name {foldername} doesn't exsit!"