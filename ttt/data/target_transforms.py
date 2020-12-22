"""
Defines all transforms to be applied on a label
"""
from typing import List, Any, Union, Tuple, Callable, Optional
from ttt.factory import Factory


class ClassificationTargetTransformer:
    """
    Transforms the input label to the appropriate target value
    for single-label classification.
    :param classes: list of relevant classes for classification
    :type classes: List[Union[str, int, float]]
    """
    def __init__(self, classes: List[Union[str, int, float]]):
        assert isinstance(classes, list)
        self.classes = classes

    def __call__(self, target:  List[Union[str, int, float]]) -> int:
        # find the intersection between target and self.classes
        intersection = [
            _target for _target in target if _target in self.classes
        ]

        # ensure that only one of the relevant classes is present
        # in the target at max
        if len(intersection) > 1:
            raise ValueError(
                'target contains more than 1 overlapping class with self.classes')

        # if one intersection, then return the corresponding index
        if len(intersection) == 1:
            return self.classes.index(intersection[0])

        raise ValueError(
            'target contains has no overlapping class with self.classes')


annotation_factory = Factory()
annotation_factory.register_builder(
    "classification", ClassificationTargetTransformer)


if __name__ == '__main__':
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    target_transformer = ClassificationTargetTransformer(classes=classes)

    sample_label = [1.0]
    label = target_transformer(sample_label)

    assert label == 1
