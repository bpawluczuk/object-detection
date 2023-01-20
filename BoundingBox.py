
class BoundingBox:

    def get_middle_point(self, x_pos, y_pos, w_len, h_len):
        return int(x_pos + w_len / 2), int(y_pos + h_len / 2)

    def bounding_boxes_merge(self, bounding_boxes, offset_x=0, offset_y=0):
        results = []
        for box in enumerate(bounding_boxes):
            x_box, y_box, w_box, h_box = box[1]
            xm, ym = self.get_middle_point(x_box, y_box, w_box, h_box)
            inside = False
            for result in enumerate(results):
                x_res, y_res, w_res, h_res = result[1]
                xmr, ymr = self.get_middle_point(x_res, y_res, w_res, h_res)
                if xmr - offset_x < xm < xmr + offset_x and ymr - offset_y < ym < ymr + offset_y:
                    inside = True
            if not inside:
                results.append(box[1])

        return results
