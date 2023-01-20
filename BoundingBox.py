def get_middle_point(x_pos, y_pos, w_len, h_len):
    return int(x_pos + w_len / 2), int(y_pos + h_len / 2)


def merge_boxes(
        boxes,
        offset_x=0,
        offset_y=0,
        origin_w=0,
        origin_h=0,
        origin_offset_w=0,
        origin_offset_h=0
):
    results = []
    for i_box, box in enumerate(boxes):

        x_box, y_box, w_box, h_box = box
        xm, ym = get_middle_point(x_box, y_box, w_box, h_box)

        inside = False

        for i_res, result in enumerate(results):

            x_res, y_res, w_res, h_res = result
            xmr, ymr = get_middle_point(x_res, y_res, w_res, h_res)

            if xmr - offset_x < xm < xmr + offset_x and ymr - offset_y < ym < ymr + offset_y:
                inside = True

        if not inside:
            results.append(box)

    return results


class BoundingBox:
    pass
