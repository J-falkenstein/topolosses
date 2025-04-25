# from .src.idgenerator import (
#     generate_password,
#     generate_guid,
#     generate_object_id,
#     generate_pin_number,
#     generate_credit_card_number,
# )
# TODO does it make sense to make something like this to have it easier to import the class?
# the above is used to make it easier to import functions

from .src.dice_loss import DiceLoss

__all__ = ["DiceLoss"]
