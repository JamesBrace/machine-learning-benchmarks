"use strict";
/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
Object.defineProperty(exports, "__esModule", { value: true });
var BoundingBox = /** @class */ (function () {
    function BoundingBox(x, y, w, h, conf, probs) {
        this.maxProb = -1;
        this.maxIndx = -1;
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.c = conf;
        this.probs = probs;
    }
    BoundingBox.prototype.getMaxProb = function () {
        if (this.maxProb === -1) {
            this.maxProb = this.probs.reduce(function (a, b) { return Math.max(a, b); });
        }
        return this.maxProb;
    };
    BoundingBox.prototype.getLabel = function () {
        if (this.maxIndx === -1) {
            this.maxIndx = this.probs.indexOf(this.getMaxProb());
        }
        return BoundingBox.LABELS[this.maxIndx];
    };
    BoundingBox.prototype.getColor = function () {
        if (this.maxIndx === -1) {
            this.maxIndx = this.probs.indexOf(this.getMaxProb());
        }
        return BoundingBox.COLORS[this.maxIndx];
    };
    BoundingBox.prototype.iou = function (box) {
        var intersection = this.intersect(box);
        var union = this.w * this.h + box.w * box.h - intersection;
        return intersection / union;
    };
    BoundingBox.prototype.intersect = function (box) {
        var width = this.overlap([this.x - this.w / 2, this.x + this.w / 2], [box.x - box.w / 2, box.x + box.w / 2]);
        var height = this.overlap([this.y - this.h / 2, this.y + this.h / 2], [box.y - box.h / 2, box.y + box.h / 2]);
        return width * height;
    };
    BoundingBox.prototype.overlap = function (intervalA, intervalB) {
        var x1 = intervalA[0], x2 = intervalA[1];
        var x3 = intervalB[0], x4 = intervalB[1];
        if (x3 < x1) {
            if (x4 < x1) {
                return 0;
            }
            else {
                return Math.min(x2, x4) - x1;
            }
        }
        else {
            if (x2 < x3) {
                return 0;
            }
            else {
                return Math.min(x2, x4) - x3;
            }
        }
    };
    BoundingBox.LABELS = ['raccoon'];
    BoundingBox.COLORS = ['rgb(43,206,72)'];
    return BoundingBox;
}());
exports.BoundingBox = BoundingBox;
