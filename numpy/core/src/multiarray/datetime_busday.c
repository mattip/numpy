/*
 * This file implements business day functionality for NumPy datetime.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/arrayobject.h>

#include "npy_config.h"
#include "npy_pycompat.h"

#include "numpy/arrayscalars.h"
#include "lowlevel_strided_loops.h"
#include "_datetime.h"
#include "datetime_busday.h"
#include "datetime_busdaycal.h"

/* Gets the day of the week for a datetime64[D] value */
static int
get_day_of_week(npy_datetime date)
{
    int day_of_week;

    /* Get the day of the week for 'date' (1970-01-05 is Monday) */
    day_of_week = (int)((date - 4) % 7);
    if (day_of_week < 0) {
        day_of_week += 7;
    }

    return day_of_week;
}

/*
 * Returns 1 if the date is a holiday (contained in the sorted
 * list of dates), 0 otherwise.
 *
 * The holidays list should be normalized, which means any NaT (not-a-time)
 * values, duplicates, and dates already excluded by the weekmask should
 * be removed, and the list should be sorted.
 */
static int
is_holiday(npy_datetime date,
            npy_datetime *holidays_begin, const npy_datetime *holidays_end)
{
    npy_datetime *trial;

    /* Simple binary search */
    while (holidays_begin < holidays_end) {
        trial = holidays_begin + (holidays_end - holidays_begin) / 2;

        if (date < *trial) {
            holidays_end = trial;
        }
        else if (date > *trial) {
            holidays_begin = trial + 1;
        }
        else {
            return 1;
        }
    }

    /* Not found */
    return 0;
}

/*
 * Finds the earliest holiday which is on or after 'date'. If 'date' does not
 * appear within the holiday range, returns 'holidays_begin' if 'date'
 * is before all holidays, or 'holidays_end' if 'date' is after all
 * holidays.
 *
 * To remove all the holidays before 'date' from a holiday range, do:
 *
 *      holidays_begin = find_holiday_earliest_on_or_after(date,
 *                                          holidays_begin, holidays_end);
 *
 * The holidays list should be normalized, which means any NaT (not-a-time)
 * values, duplicates, and dates already excluded by the weekmask should
 * be removed, and the list should be sorted.
 */
static npy_datetime *
find_earliest_holiday_on_or_after(npy_datetime date,
            npy_datetime *holidays_begin, const npy_datetime *holidays_end)
{
    npy_datetime *trial;

    /* Simple binary search */
    while (holidays_begin < holidays_end) {
        trial = holidays_begin + (holidays_end - holidays_begin) / 2;

        if (date < *trial) {
            holidays_end = trial;
        }
        else if (date > *trial) {
            holidays_begin = trial + 1;
        }
        else {
            return trial;
        }
    }

    return holidays_begin;
}

/*
 * Finds the earliest holiday which is after 'date'. If 'date' does not
 * appear within the holiday range, returns 'holidays_begin' if 'date'
 * is before all holidays, or 'holidays_end' if 'date' is after all
 * holidays.
 *
 * To remove all the holidays after 'date' from a holiday range, do:
 *
 *      holidays_end = find_holiday_earliest_after(date,
 *                                          holidays_begin, holidays_end);
 *
 * The holidays list should be normalized, which means any NaT (not-a-time)
 * values, duplicates, and dates already excluded by the weekmask should
 * be removed, and the list should be sorted.
 */
static npy_datetime *
find_earliest_holiday_after(npy_datetime date,
            npy_datetime *holidays_begin, const npy_datetime *holidays_end)
{
    npy_datetime *trial;

    /* Simple binary search */
    while (holidays_begin < holidays_end) {
        trial = holidays_begin + (holidays_end - holidays_begin) / 2;

        if (date < *trial) {
            holidays_end = trial;
        }
        else if (date > *trial) {
            holidays_begin = trial + 1;
        }
        else {
            return trial + 1;
        }
    }

    return holidays_begin;
}

/*
 * Applies the 'roll' strategy to 'date', placing the result in 'out'
 * and setting 'out_day_of_week' to the day of the week that results.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
apply_business_day_roll(HPyContext *ctx, npy_datetime date, npy_datetime *out,
                    int *out_day_of_week,
                    NPY_BUSDAY_ROLL roll,
                    const npy_bool *weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    int day_of_week;

    /* Deal with NaT input */
    if (date == NPY_DATETIME_NAT) {
        *out = NPY_DATETIME_NAT;
        if (roll == NPY_BUSDAY_RAISE) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "NaT input in busday_offset");
            return -1;
        }
        else {
            return 0;
        }
    }

    /* Get the day of the week for 'date' */
    day_of_week = get_day_of_week(date);

    /* Apply the 'roll' if it's not a business day */
    if (weekmask[day_of_week] == 0 ||
                        is_holiday(date, holidays_begin, holidays_end)) {
        npy_datetime start_date = date;
        int start_day_of_week = day_of_week;

        switch (roll) {
            case NPY_BUSDAY_FOLLOWING:
            case NPY_BUSDAY_MODIFIEDFOLLOWING: {
                do {
                    ++date;
                    if (++day_of_week == 7) {
                        day_of_week = 0;
                    }
                } while (weekmask[day_of_week] == 0 ||
                            is_holiday(date, holidays_begin, holidays_end));

                if (roll == NPY_BUSDAY_MODIFIEDFOLLOWING) {
                    /* If we crossed a month boundary, do preceding instead */
                    if (days_to_month_number(start_date) !=
                                days_to_month_number(date)) {
                        date = start_date;
                        day_of_week = start_day_of_week;

                        do {
                            --date;
                            if (--day_of_week == -1) {
                                day_of_week = 6;
                            }
                        } while (weekmask[day_of_week] == 0 ||
                            is_holiday(date, holidays_begin, holidays_end));
                    }
                }
                break;
            }
            case NPY_BUSDAY_PRECEDING:
            case NPY_BUSDAY_MODIFIEDPRECEDING: {
                do {
                    --date;
                    if (--day_of_week == -1) {
                        day_of_week = 6;
                    }
                } while (weekmask[day_of_week] == 0 ||
                            is_holiday(date, holidays_begin, holidays_end));

                if (roll == NPY_BUSDAY_MODIFIEDPRECEDING) {
                    /* If we crossed a month boundary, do following instead */
                    if (days_to_month_number(start_date) !=
                                days_to_month_number(date)) {
                        date = start_date;
                        day_of_week = start_day_of_week;

                        do {
                            ++date;
                            if (++day_of_week == 7) {
                                day_of_week = 0;
                            }
                        } while (weekmask[day_of_week] == 0 ||
                            is_holiday(date, holidays_begin, holidays_end));
                    }
                }
                break;
            }
            case NPY_BUSDAY_NAT: {
                date = NPY_DATETIME_NAT;
                break;
            }
            case NPY_BUSDAY_RAISE: {
                *out = NPY_DATETIME_NAT;
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "Non-business day date in busday_offset");
                return -1;
            }
        }
    }

    *out = date;
    *out_day_of_week = day_of_week;

    return 0;
}

/*
 * Applies a single business day offset. See the function
 * business_day_offset for the meaning of all the parameters.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
apply_business_day_offset(HPyContext *ctx, npy_datetime date, npy_int64 offset,
                    npy_datetime *out,
                    NPY_BUSDAY_ROLL roll,
                    npy_bool *weekmask, int busdays_in_weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    int day_of_week = 0;
    npy_datetime *holidays_temp;

    /* Roll the date to a business day */
    if (apply_business_day_roll(ctx, date, &date, &day_of_week,
                                roll,
                                weekmask,
                                holidays_begin, holidays_end) < 0) {
        return -1;
    }

    /* If we get a NaT, just return it */
    if (date == NPY_DATETIME_NAT) {
        *out = NPY_DATETIME_NAT;
        return 0;
    }

    /* Now we're on a valid business day */
    if (offset > 0) {
        /* Remove any earlier holidays */
        holidays_begin = find_earliest_holiday_on_or_after(date,
                                            holidays_begin, holidays_end);

        /* Jump by as many weeks as we can */
        date += (offset / busdays_in_weekmask) * 7;
        offset = offset % busdays_in_weekmask;

        /* Adjust based on the number of holidays we crossed */
        holidays_temp = find_earliest_holiday_after(date,
                                            holidays_begin, holidays_end);
        offset += holidays_temp - holidays_begin;
        holidays_begin = holidays_temp;

        /* Step until we use up the rest of the offset */
        while (offset > 0) {
            ++date;
            if (++day_of_week == 7) {
                day_of_week = 0;
            }
            if (weekmask[day_of_week] && !is_holiday(date,
                                            holidays_begin, holidays_end)) {
                offset--;
            }
        }
    }
    else if (offset < 0) {
        /* Remove any later holidays */
        holidays_end = find_earliest_holiday_after(date,
                                            holidays_begin, holidays_end);

        /* Jump by as many weeks as we can */
        date += (offset / busdays_in_weekmask) * 7;
        offset = offset % busdays_in_weekmask;

        /* Adjust based on the number of holidays we crossed */
        holidays_temp = find_earliest_holiday_on_or_after(date,
                                            holidays_begin, holidays_end);
        offset -= holidays_end - holidays_temp;
        holidays_end = holidays_temp;

        /* Step until we use up the rest of the offset */
        while (offset < 0) {
            --date;
            if (--day_of_week == -1) {
                day_of_week = 6;
            }
            if (weekmask[day_of_week] && !is_holiday(date,
                                            holidays_begin, holidays_end)) {
                offset++;
            }
        }
    }

    *out = date;
    return 0;
}

/*
 * Applies a single business day count operation. See the function
 * business_day_count for the meaning of all the parameters.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
apply_business_day_count(HPyContext *ctx, npy_datetime date_begin, npy_datetime date_end,
                    npy_int64 *out,
                    const npy_bool *weekmask, int busdays_in_weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    npy_int64 count, whole_weeks;
    int day_of_week = 0;
    int swapped = 0;

    /* If we get a NaT, raise an error */
    if (date_begin == NPY_DATETIME_NAT || date_end == NPY_DATETIME_NAT) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Cannot compute a business day count with a NaT (not-a-time) "
                "date");
        return -1;
    }

    /* Trivial empty date range */
    if (date_begin == date_end) {
        *out = 0;
        return 0;
    }
    else if (date_begin > date_end) {
        npy_datetime tmp = date_begin;
        date_begin = date_end;
        date_end = tmp;
        swapped = 1;
    }

    /* Remove any earlier holidays */
    holidays_begin = find_earliest_holiday_on_or_after(date_begin,
                                        holidays_begin, holidays_end);
    /* Remove any later holidays */
    holidays_end = find_earliest_holiday_on_or_after(date_end,
                                        holidays_begin, holidays_end);

    /* Start the count as negative the number of holidays in the range */
    count = -(holidays_end - holidays_begin);

    /* Add the whole weeks between date_begin and date_end */
    whole_weeks = (date_end - date_begin) / 7;
    count += whole_weeks * busdays_in_weekmask;
    date_begin += whole_weeks * 7;

    if (date_begin < date_end) {
        /* Get the day of the week for 'date_begin' */
        day_of_week = get_day_of_week(date_begin);

        /* Count the remaining days one by one */
        while (date_begin < date_end) {
            if (weekmask[day_of_week]) {
                count++;
            }
            ++date_begin;
            if (++day_of_week == 7) {
                day_of_week = 0;
            }
        }
    }

    if (swapped) {
        count = -count;
    }

    *out = count;
    return 0;
}

/*
 * Applies the given offsets in business days to the dates provided.
 * This is the low-level function which requires already cleaned input
 * data.
 *
 * dates:    An array of dates with 'datetime64[D]' data type.
 * offsets:  An array safely convertible into type int64.
 * out:      Either NULL, or an array with 'datetime64[D]' data type
 *              in which to place the resulting dates.
 * roll:     A rule for how to treat non-business day dates.
 * weekmask: A 7-element boolean mask, 1 for possible business days and 0
 *              for non-business days.
 * busdays_in_weekmask: A count of how many 1's there are in weekmask.
 * holidays_begin/holidays_end: A sorted list of dates matching '[D]'
 *           unit metadata, with any dates falling on a day of the
 *           week without weekmask[i] == 1 already filtered out.
 *
 * For each (date, offset) in the broadcasted pair of (dates, offsets),
 * does the following:
 *  + Applies the 'roll' rule to the date to either produce NaT, raise
 *    an exception, or land on a valid business day.
 *  + Adds 'offset' business days to the valid business day found.
 *  + Sets the value in 'out' if provided, or the allocated output array
 *    otherwise.
 */
NPY_NO_EXPORT HPy // PyArrayObject *
business_day_offset(HPyContext *ctx, 
                    HPy /* PyArrayObject * */ dates, 
                    HPy /* PyArrayObject * */ offsets,
                    HPy /* PyArrayObject * */ out,
                    NPY_BUSDAY_ROLL roll,
                    npy_bool *weekmask, int busdays_in_weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    PyArray_DatetimeMetaData temp_meta;
    HPy dtypes[3] = {HPy_NULL, HPy_NULL, HPy_NULL}; // PyArray_Descr *

    NpyIter *iter = NULL;
    HPy op[3] = {HPy_NULL, HPy_NULL, HPy_NULL}; // PyArrayObject *
    npy_uint32 op_flags[3], flags;

    HPy ret = HPy_NULL; // PyArrayObject *

    if (busdays_in_weekmask == 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "the business day weekmask must have at least one "
                "valid business day");
        return HPy_NULL;
    }

    /* First create the data types for dates and offsets */
    temp_meta.base = NPY_FR_D;
    temp_meta.num = 1;
    dtypes[0] = hpy_create_datetime_dtype(ctx, NPY_DATETIME, &temp_meta);
    if (HPy_IsNull(dtypes[0])) {
        goto fail;
    }
    dtypes[1] = HPyArray_DescrFromType(ctx, NPY_INT64);
    if (HPy_IsNull(dtypes[1])) {
        goto fail;
    }
    dtypes[2] = HPy_Dup(ctx, dtypes[0]);
    // Py_INCREF(dtypes[2]);

    /* Set up the iterator parameters */
    flags = NPY_ITER_EXTERNAL_LOOP|
            NPY_ITER_BUFFERED|
            NPY_ITER_ZEROSIZE_OK;
    op[0] = dates;
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op[1] = offsets;
    op_flags[1] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op[2] = out;
    op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_ALIGNED;

    /* Allocate the iterator */
    iter = HNpyIter_MultiNew(ctx, 3, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, dtypes);
    if (iter == NULL) {
        goto fail;
    }

    /* Loop over all elements */
    if (NpyIter_GetIterSize(iter) > 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strideptr, *innersizeptr;

        iternext = HNpyIter_GetIterNext(ctx, iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        strideptr = NpyIter_GetInnerStrideArray(iter);
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

        do {
            char *data_dates = dataptr[0];
            char *data_offsets = dataptr[1];
            char *data_out = dataptr[2];
            npy_intp stride_dates = strideptr[0];
            npy_intp stride_offsets = strideptr[1];
            npy_intp stride_out = strideptr[2];
            npy_intp count = *innersizeptr;

            while (count--) {
                if (apply_business_day_offset(ctx, *(npy_int64 *)data_dates,
                                       *(npy_int64 *)data_offsets,
                                       (npy_int64 *)data_out,
                                       roll,
                                       weekmask, busdays_in_weekmask,
                                       holidays_begin, holidays_end) < 0) {
                    goto fail;
                }

                data_dates += stride_dates;
                data_offsets += stride_offsets;
                data_out += stride_out;
            }
        } while (iternext(ctx, iter));
    }

    /* Get the return object from the iterator */
    ret = HPy_Dup(ctx, HNpyIter_GetOperandArray(iter)[2]);
    // Py_INCREF(ret);

    goto finish;

fail:
    HPy_Close(ctx, ret);
    ret = HPy_NULL;

finish:
    HPy_Close(ctx, dtypes[0]);
    HPy_Close(ctx, dtypes[1]);
    HPy_Close(ctx, dtypes[2]);
    if (iter != NULL) {
        if (HNpyIter_Deallocate(ctx, iter) != NPY_SUCCEED) {
            HPy_Close(ctx, ret);
            ret = HPy_NULL;
        }
    }
    return ret;
}

/*
 * Counts the number of business days between two dates, not including
 * the end date. This is the low-level function which requires already
 * cleaned input data.
 *
 * If dates_begin is before dates_end, the result is positive.  If
 * dates_begin is after dates_end, it is negative.
 *
 * dates_begin:  An array of dates with 'datetime64[D]' data type.
 * dates_end:    An array of dates with 'datetime64[D]' data type.
 * out:      Either NULL, or an array with 'int64' data type
 *              in which to place the resulting dates.
 * weekmask: A 7-element boolean mask, 1 for possible business days and 0
 *              for non-business days.
 * busdays_in_weekmask: A count of how many 1's there are in weekmask.
 * holidays_begin/holidays_end: A sorted list of dates matching '[D]'
 *           unit metadata, with any dates falling on a day of the
 *           week without weekmask[i] == 1 already filtered out.
 */
NPY_NO_EXPORT HPy // PyArrayObject *
business_day_count(HPyContext *ctx, HPy /* PyArrayObject * */ dates_begin, 
                    HPy /* PyArrayObject * */ dates_end,
                    HPy /* PyArrayObject * */ out,
                    npy_bool *weekmask, int busdays_in_weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    PyArray_DatetimeMetaData temp_meta;
    HPy dtypes[3] = {HPy_NULL, HPy_NULL, HPy_NULL}; // PyArray_Descr *

    NpyIter *iter = NULL;
    HPy op[3] = {HPy_NULL, HPy_NULL, HPy_NULL}; // PyArrayObject *
    npy_uint32 op_flags[3], flags;

    HPy ret = HPy_NULL; // PyArrayObject *

    if (busdays_in_weekmask == 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "the business day weekmask must have at least one "
                "valid business day");
        return HPy_NULL;
    }

    /* First create the data types for the dates and the int64 output */
    temp_meta.base = NPY_FR_D;
    temp_meta.num = 1;
    dtypes[0] = hpy_create_datetime_dtype(ctx, NPY_DATETIME, &temp_meta);
    if (HPy_IsNull(dtypes[0])) {
        goto fail;
    }
    dtypes[1] = HPy_Dup(ctx, dtypes[0]);
    // Py_INCREF(dtypes[1]);
    dtypes[2] = HPyArray_DescrFromType(ctx, NPY_INT64);
    if (HPy_IsNull(dtypes[2])) {
        goto fail;
    }

    /* Set up the iterator parameters */
    flags = NPY_ITER_EXTERNAL_LOOP|
            NPY_ITER_BUFFERED|
            NPY_ITER_ZEROSIZE_OK;
    op[0] = dates_begin;
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op[1] = dates_end;
    op_flags[1] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op[2] = out;
    op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_ALIGNED;

    /* Allocate the iterator */
    iter = HNpyIter_MultiNew(ctx, 3, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, dtypes);
    if (iter == NULL) {
        goto fail;
    }

    /* Loop over all elements */
    if (NpyIter_GetIterSize(iter) > 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strideptr, *innersizeptr;

        iternext = HNpyIter_GetIterNext(ctx, iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        strideptr = NpyIter_GetInnerStrideArray(iter);
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

        do {
            char *data_dates_begin = dataptr[0];
            char *data_dates_end = dataptr[1];
            char *data_out = dataptr[2];
            npy_intp stride_dates_begin = strideptr[0];
            npy_intp stride_dates_end = strideptr[1];
            npy_intp stride_out = strideptr[2];
            npy_intp count = *innersizeptr;

            while (count--) {
                if (apply_business_day_count(ctx, *(npy_int64 *)data_dates_begin,
                                       *(npy_int64 *)data_dates_end,
                                       (npy_int64 *)data_out,
                                       weekmask, busdays_in_weekmask,
                                       holidays_begin, holidays_end) < 0) {
                    goto fail;
                }

                data_dates_begin += stride_dates_begin;
                data_dates_end += stride_dates_end;
                data_out += stride_out;
            }
        } while (iternext(ctx, iter));
    }

    /* Get the return object from the iterator */
    ret = HPy_Dup(ctx, HNpyIter_GetOperandArray(iter)[2]);

    goto finish;

fail:
    HPy_Close(ctx, ret);
    ret = HPy_NULL;

finish:
    HPy_Close(ctx, dtypes[0]);
    HPy_Close(ctx, dtypes[1]);
    HPy_Close(ctx, dtypes[2]);
    if (iter != NULL) {
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            HPy_Close(ctx, ret);
            ret = HPy_NULL;
        }
    }
    return ret;
}

/*
 * Returns a boolean array with True for input dates which are valid
 * business days, and False for dates which are not. This is the
 * low-level function which requires already cleaned input data.
 *
 * dates:  An array of dates with 'datetime64[D]' data type.
 * out:      Either NULL, or an array with 'bool' data type
 *              in which to place the resulting dates.
 * weekmask: A 7-element boolean mask, 1 for possible business days and 0
 *              for non-business days.
 * busdays_in_weekmask: A count of how many 1's there are in weekmask.
 * holidays_begin/holidays_end: A sorted list of dates matching '[D]'
 *           unit metadata, with any dates falling on a day of the
 *           week without weekmask[i] == 1 already filtered out.
 */
NPY_NO_EXPORT HPy // PyArrayObject *
is_business_day(HPyContext *ctx, HPy /* PyArrayObject * */ dates, 
                    HPy /* PyArrayObject * */ out,
                    const npy_bool *weekmask, int busdays_in_weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    PyArray_DatetimeMetaData temp_meta;
    HPy dtypes[2] = {HPy_NULL, HPy_NULL}; // PyArray_Descr *

    NpyIter *iter = NULL;
    HPy op[2] = {HPy_NULL, HPy_NULL}; // PyArrayObject *
    npy_uint32 op_flags[2], flags;

    HPy ret = HPy_NULL; // PyArrayObject *

    if (busdays_in_weekmask == 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "the business day weekmask must have at least one "
                "valid business day");
        return HPy_NULL;
    }

    /* First create the data types for the dates and the bool output */
    temp_meta.base = NPY_FR_D;
    temp_meta.num = 1;
    dtypes[0] = hpy_create_datetime_dtype(ctx, NPY_DATETIME, &temp_meta);
    if (HPy_IsNull(dtypes[0])) {
        goto fail;
    }
    dtypes[1] = HPyArray_DescrFromType(ctx, NPY_BOOL);
    if (HPy_IsNull(dtypes[1])) {
        goto fail;
    }

    /* Set up the iterator parameters */
    flags = NPY_ITER_EXTERNAL_LOOP|
            NPY_ITER_BUFFERED|
            NPY_ITER_ZEROSIZE_OK;
    op[0] = dates;
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op[1] = out;
    op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_ALIGNED;

    /* Allocate the iterator */
    iter = HNpyIter_MultiNew(ctx, 2, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, dtypes);
    if (iter == NULL) {
        goto fail;
    }

    /* Loop over all elements */
    if (NpyIter_GetIterSize(iter) > 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strideptr, *innersizeptr;

        iternext = HNpyIter_GetIterNext(ctx, iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        strideptr = NpyIter_GetInnerStrideArray(iter);
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

        do {
            char *data_dates = dataptr[0];
            char *data_out = dataptr[1];
            npy_intp stride_dates = strideptr[0];
            npy_intp stride_out = strideptr[1];
            npy_intp count = *innersizeptr;

            npy_datetime date;
            int day_of_week;

            while (count--) {
                /* Check if it's a business day */
                date = *(npy_datetime *)data_dates;
                day_of_week = get_day_of_week(date);
                *(npy_bool *)data_out = weekmask[day_of_week] &&
                                        !is_holiday(date,
                                            holidays_begin, holidays_end) &&
                                        date != NPY_DATETIME_NAT;

                data_dates += stride_dates;
                data_out += stride_out;
            }
        } while (iternext(ctx, iter));
    }

    /* Get the return object from the iterator */
    ret = HPy_Dup(ctx, HNpyIter_GetOperandArray(iter)[1]);
    // Py_INCREF(ret);

    goto finish;

fail:
    HPy_Close(ctx, ret);
    ret = HPy_NULL;

finish:
    HPy_Close(ctx, dtypes[0]);
    HPy_Close(ctx, dtypes[1]);
    if (iter != NULL) {
        if (HNpyIter_Deallocate(ctx, iter) != NPY_SUCCEED) {
            HPy_Close(ctx, ret);
            ret = HPy_NULL;
        }
    }
    return ret;
}

static int
HPyArray_BusDayRollConverter(HPyContext *ctx, HPy roll_in, NPY_BUSDAY_ROLL *roll)
{
    HPy obj = roll_in;

    /* Make obj into an UTF8 string */
    if (HPyBytes_Check(ctx, obj)) {
        /* accept bytes input */
        HPy obj_str = HPyUnicode_FromEncodedObject(ctx, obj, NULL, NULL);
        if (HPy_IsNull(obj_str)) {
            return 0;
        }
        obj = obj_str;
    }
    else {
        obj = HPy_Dup(ctx, obj);
    }

    Py_ssize_t len;
    char const *str = HPyUnicode_AsUTF8AndSize(ctx, obj, &len);
    if (str == NULL) {
        HPy_Close(ctx, obj);
        return 0;
    }

    /* Use switch statements to quickly isolate the right enum value */
    switch (str[0]) {
        case 'b':
            if (strcmp(str, "backward") == 0) {
                *roll = NPY_BUSDAY_BACKWARD;
                goto finish;
            }
            break;
        case 'f':
            if (len > 2) switch (str[2]) {
                case 'r':
                    if (strcmp(str, "forward") == 0) {
                        *roll = NPY_BUSDAY_FORWARD;
                        goto finish;
                    }
                    break;
                case 'l':
                    if (strcmp(str, "following") == 0) {
                        *roll = NPY_BUSDAY_FOLLOWING;
                        goto finish;
                    }
                    break;
            }
            break;
        case 'm':
            if (len > 8) switch (str[8]) {
                case 'f':
                    if (strcmp(str, "modifiedfollowing") == 0) {
                        *roll = NPY_BUSDAY_MODIFIEDFOLLOWING;
                        goto finish;
                    }
                    break;
                case 'p':
                    if (strcmp(str, "modifiedpreceding") == 0) {
                        *roll = NPY_BUSDAY_MODIFIEDPRECEDING;
                        goto finish;
                    }
                    break;
            }
            break;
        case 'n':
            if (strcmp(str, "nat") == 0) {
                *roll = NPY_BUSDAY_NAT;
                goto finish;
            }
            break;
        case 'p':
            if (strcmp(str, "preceding") == 0) {
                *roll = NPY_BUSDAY_PRECEDING;
                goto finish;
            }
            break;
        case 'r':
            if (strcmp(str, "raise") == 0) {
                *roll = NPY_BUSDAY_RAISE;
                goto finish;
            }
            break;
    }

    HPyErr_Format_p(ctx, ctx->h_ValueError,
            "Invalid business day roll parameter \"%s\"",
            str);
    HPy_Close(ctx, obj);
    return 0;

finish:
    HPy_Close(ctx, obj);
    return 1;
}

/*
 * This is the 'busday_offset' function exposed for calling
 * from Python.
 */
HPyDef_METH(array_busday_offset, "busday_offset", array_busday_offset_impl, HPyFunc_KEYWORDS)
NPY_NO_EXPORT HPy
array_busday_offset_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    static char *kwlist[] = {"dates", "offsets", "roll",
                             "weekmask", "holidays", "busdaycal", "out", NULL};

    HPy dates_in = HPy_NULL, offsets_in = HPy_NULL, out_in = HPy_NULL;

    HPy dates = HPy_NULL, offsets = HPy_NULL, out = HPy_NULL, ret; // PyArrayObject *
    NPY_BUSDAY_ROLL roll = NPY_BUSDAY_RAISE;
    npy_bool weekmask[7] = {2, 1, 1, 1, 1, 0, 0};
    NpyBusDayCalendar *busdaycal = NULL;
    int i, busdays_in_weekmask;
    npy_holidayslist holidays = {NULL, NULL};
    int allocated_holidays = 1;

    HPy h_weekmask_0 = HPy_NULL, h_holiday = HPy_NULL, h_busdaycal = HPy_NULL, h_roll = HPy_NULL;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds,
                                    "OO|O&O&O&O!O:busday_offset", kwlist,
                                    &dates_in,
                                    &offsets_in,
                                    &roll,
                                    &weekmask[0],
                                    &holidays,
                                    &busdaycal,
                                    &out_in)) {
        return HPy_NULL;
    }
    if (HPyArray_BusDayRollConverter(ctx, h_roll, &roll) != NPY_SUCCEED ||
            HPyArray_WeekMaskConverter(ctx, h_weekmask_0, &weekmask[0]) != NPY_SUCCEED ||
            HPyArray_HolidaysConverter(ctx, h_holiday, &holidays) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "busday_offset: TODO");
        goto fail;
    }

    HPy HNpyBusDayCalendar_Type = HPy_FromPyObject(ctx, (PyObject *)&NpyBusDayCalendar_Type);
    if (!HPy_TypeCheck(ctx, h_busdaycal, HNpyBusDayCalendar_Type)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "busday_offset: TODO");
        HPy_Close(ctx, HNpyBusDayCalendar_Type);
        goto fail;
    }
    HPy_Close(ctx, HNpyBusDayCalendar_Type);
    busdaycal = (NpyBusDayCalendar *)HPy_AsPyObject(ctx, h_busdaycal);


    /* Make sure only one of the weekmask/holidays and busdaycal is supplied */
    if (busdaycal != NULL) {
        if (weekmask[0] != 2 || holidays.begin != NULL) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Cannot supply both the weekmask/holidays and the "
                    "busdaycal parameters to busday_offset()");
            goto fail;
        }

        /* Indicate that the holidays weren't allocated by us */
        allocated_holidays = 0;

        /* Copy the private normalized weekmask/holidays data */
        holidays = busdaycal->holidays;
        busdays_in_weekmask = busdaycal->busdays_in_weekmask;
        memcpy(weekmask, busdaycal->weekmask, 7);
        Py_DECREF(busdaycal);
    }
    else {
        /*
         * Fix up the weekmask from the uninitialized
         * signal value to a proper default.
         */
        if (weekmask[0] == 2) {
            weekmask[0] = 1;
        }

        /* Count the number of business days in a week */
        busdays_in_weekmask = 0;
        for (i = 0; i < 7; ++i) {
            busdays_in_weekmask += weekmask[i];
        }

        /* The holidays list must be normalized before using it */
        normalize_holidays_list(&holidays, weekmask);
    }

    /* Make 'dates' into an array */
    if (HPyArray_Check(ctx, dates_in)) {
        dates = HPy_Dup(ctx, dates_in);
        // Py_INCREF(dates);
    }
    else {
        HPy datetime_dtype; // PyArray_Descr *

        /* Use the datetime dtype with generic units so it fills it in */
        datetime_dtype = HPyArray_DescrFromType(ctx, NPY_DATETIME);
        if (HPy_IsNull(datetime_dtype)) {
            goto fail;
        }

        /* This steals the datetime_dtype reference */
        dates = HPyArray_FromAny(ctx, dates_in, datetime_dtype,
                                                0, 0, 0, HPy_NULL);
        if (HPy_IsNull(dates)) {
            goto fail;
        }
    }

    /* Make 'offsets' into an array */
    HPy npyint64_descr = HPyArray_DescrFromType(ctx, NPY_INT64);
    offsets = HPyArray_FromAny(ctx, offsets_in,
                            npyint64_descr,
                            0, 0, 0, HPy_NULL);
    if (HPy_IsNull(offsets)) {
        goto fail;
    }

    /* Make sure 'out' is an array if it's provided */
    if (!HPy_IsNull(out_in)) {
        if (!HPyArray_Check(ctx, out_in)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "busday_offset: must provide a NumPy array for 'out'");
            goto fail;
        }
        out = out_in;
    }

    ret = business_day_offset(ctx, dates, offsets, out, roll,
                    weekmask, busdays_in_weekmask,
                    holidays.begin, holidays.end);

    HPy_Close(ctx, dates);
    HPy_Close(ctx, offsets);
    if (allocated_holidays && holidays.begin != NULL) {
        PyArray_free(holidays.begin);
    }

    if (HPy_IsNull(out)) {
        out = HPyArray_Return(ctx, ret);
        HPy_Close(ctx, ret);
        ret = out;
    }
    return ret;

fail:
    HPy_Close(ctx, dates);
    HPy_Close(ctx, offsets);
    if (allocated_holidays && holidays.begin != NULL) {
        PyArray_free(holidays.begin);
    }

    return HPy_NULL;
}

/*
 * This is the 'busday_count' function exposed for calling
 * from Python.
 */
HPyDef_METH(array_busday_count, "busday_count", array_busday_count_impl, HPyFunc_KEYWORDS)
NPY_NO_EXPORT HPy
array_busday_count_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    static char *kwlist[] = {"begindates", "enddates",
                             "weekmask", "holidays", "busdaycal", "out", NULL};

    HPy dates_begin_in = HPy_NULL, dates_end_in = HPy_NULL, out_in = HPy_NULL;

    HPy dates_begin = HPy_NULL, dates_end = HPy_NULL, out = HPy_NULL, ret; // PyArrayObject *
    npy_bool weekmask[7] = {2, 1, 1, 1, 1, 0, 0};
    NpyBusDayCalendar *busdaycal = NULL;
    int i, busdays_in_weekmask;
    npy_holidayslist holidays = {NULL, NULL};
    int allocated_holidays = 1;

    HPy h_weekmask_0 = HPy_NULL, h_holiday = HPy_NULL, h_busdaycal = HPy_NULL;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds,
                                    "OO|O&O&O!O:busday_count", kwlist,
                                    &dates_begin_in,
                                    &dates_end_in,
                                    &weekmask[0],
                                    &holidays,
                                    &busdaycal,
                                    &out_in)) {
        return HPy_NULL;
    }
    if (HPyArray_WeekMaskConverter(ctx, h_weekmask_0, &weekmask[0]) != NPY_SUCCEED ||
            HPyArray_HolidaysConverter(ctx, h_holiday, &holidays) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "busday_count: TODO");
        goto fail;
    }

    HPy HNpyBusDayCalendar_Type = HPy_FromPyObject(ctx, (PyObject *)&NpyBusDayCalendar_Type);
    if (!HPy_TypeCheck(ctx, h_busdaycal, HNpyBusDayCalendar_Type)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "busday_count: TODO");
        HPy_Close(ctx, HNpyBusDayCalendar_Type);
        goto fail;
    }
    HPy_Close(ctx, HNpyBusDayCalendar_Type);
    busdaycal = (NpyBusDayCalendar *)HPy_AsPyObject(ctx, h_busdaycal);

    /* Make sure only one of the weekmask/holidays and busdaycal is supplied */
    if (busdaycal != NULL) {
        if (weekmask[0] != 2 || holidays.begin != NULL) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Cannot supply both the weekmask/holidays and the "
                    "busdaycal parameters to busday_count()");
            Py_DECREF(busdaycal);
            goto fail;
        }

        /* Indicate that the holidays weren't allocated by us */
        allocated_holidays = 0;

        /* Copy the private normalized weekmask/holidays data */
        holidays = busdaycal->holidays;
        busdays_in_weekmask = busdaycal->busdays_in_weekmask;
        memcpy(weekmask, busdaycal->weekmask, 7);
        Py_DECREF(busdaycal);
    }
    else {
        /*
         * Fix up the weekmask from the uninitialized
         * signal value to a proper default.
         */
        if (weekmask[0] == 2) {
            weekmask[0] = 1;
        }

        /* Count the number of business days in a week */
        busdays_in_weekmask = 0;
        for (i = 0; i < 7; ++i) {
            busdays_in_weekmask += weekmask[i];
        }

        /* The holidays list must be normalized before using it */
        normalize_holidays_list(&holidays, weekmask);
    }

    /* Make 'dates_begin' into an array */
    if (HPyArray_Check(ctx, dates_begin_in)) {
        dates_begin = HPy_Dup(ctx, dates_begin_in);
        // Py_INCREF(dates_begin);
    }
    else {
        HPy datetime_dtype; // PyArray_Descr *

        /* Use the datetime dtype with generic units so it fills it in */
        datetime_dtype = HPyArray_DescrFromType(ctx, NPY_DATETIME);
        if (HPy_IsNull(datetime_dtype)) {
            goto fail;
        }

        /* This steals the datetime_dtype reference */
        dates_begin = HPyArray_FromAny(ctx, dates_begin_in,
                                                datetime_dtype,
                                                0, 0, 0, HPy_NULL);
        if (HPy_IsNull(dates_begin)) {
            goto fail;
        }
    }

    /* Make 'dates_end' into an array */
    if (HPyArray_Check(ctx, dates_end_in)) {
        dates_end = HPy_Dup(ctx, dates_end_in);
        // Py_INCREF(dates_end);
    }
    else {
        HPy datetime_dtype; // PyArray_Descr *

        /* Use the datetime dtype with generic units so it fills it in */
        datetime_dtype = HPyArray_DescrFromType(ctx, NPY_DATETIME);
        if (HPy_IsNull(datetime_dtype)) {
            goto fail;
        }

        /* This steals the datetime_dtype reference */
        dates_end = HPyArray_FromAny(ctx, dates_end_in,
                                                datetime_dtype,
                                                0, 0, 0, HPy_NULL);
        if (HPy_IsNull(dates_end)) {
            goto fail;
        }
    }

    /* Make sure 'out' is an array if it's provided */
    if (!HPy_IsNull(out_in)) {
        if (!HPyArray_Check(ctx, out_in)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "busday_offset: must provide a NumPy array for 'out'");
            goto fail;
        }
        out = out_in;
    }

    ret = business_day_count(ctx, dates_begin, dates_end, out, 
                    weekmask, busdays_in_weekmask,
                    holidays.begin, holidays.end);

    HPy_Close(ctx, dates_begin);
    HPy_Close(ctx, dates_end);
    if (allocated_holidays && holidays.begin != NULL) {
        PyArray_free(holidays.begin);
    }
    if (HPy_IsNull(out)) {
        out = HPyArray_Return(ctx, ret);
        HPy_Close(ctx, ret);
        ret = out;
    }
    HPyTracker_Close(ctx, ht);
    return ret;

fail:
    HPyTracker_Close(ctx, ht);
    HPy_Close(ctx, dates_begin);
    HPy_Close(ctx, dates_end);
    if (allocated_holidays && holidays.begin != NULL) {
        PyArray_free(holidays.begin);
    }

    return HPy_NULL;
}

/*
 * This is the 'is_busday' function exposed for calling
 * from Python.
 */
HPyDef_METH(array_is_busday, "is_busday", array_is_busday_impl, HPyFunc_KEYWORDS)
NPY_NO_EXPORT HPy
array_is_busday_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    static char *kwlist[] = {"dates",
                             "weekmask", "holidays", "busdaycal", "out", NULL};

    HPy dates_in = HPy_NULL, out_in = HPy_NULL;

    HPy dates = HPy_NULL,out = HPy_NULL, ret; // PyArrayObject *
    npy_bool weekmask[7] = {2, 1, 1, 1, 1, 0, 0};
    NpyBusDayCalendar *busdaycal = NULL;
    int i, busdays_in_weekmask;
    npy_holidayslist holidays = {NULL, NULL};
    int allocated_holidays = 1;

    HPy h_weekmask_0 = HPy_NULL, h_holiday = HPy_NULL, h_busdaycal = HPy_NULL;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds,
                                    "O|O&O&O!O:is_busday", kwlist,
                                    &dates_in,
                                    &weekmask[0],
                                    &holidays,
                                    &busdaycal,
                                    &out_in)) {
        return HPy_NULL;
    }
    if (HPyArray_WeekMaskConverter(ctx, h_weekmask_0, &weekmask[0]) != NPY_SUCCEED ||
            HPyArray_HolidaysConverter(ctx, h_holiday, &holidays) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "is_busday: TODO");
        goto fail;
    }

    HPy HNpyBusDayCalendar_Type = HPy_FromPyObject(ctx, (PyObject *)&NpyBusDayCalendar_Type);
    if (!HPy_TypeCheck(ctx, h_busdaycal, HNpyBusDayCalendar_Type)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "is_busday: TODO");
        HPy_Close(ctx, HNpyBusDayCalendar_Type);
        goto fail;
    }
    HPy_Close(ctx, HNpyBusDayCalendar_Type);
    busdaycal = (NpyBusDayCalendar *)HPy_AsPyObject(ctx, h_busdaycal);

    /* Make sure only one of the weekmask/holidays and busdaycal is supplied */
    if (busdaycal != NULL) {
        if (weekmask[0] != 2 || holidays.begin != NULL) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Cannot supply both the weekmask/holidays and the "
                    "busdaycal parameters to is_busday()");
            goto fail;
        }

        /* Indicate that the holidays weren't allocated by us */
        allocated_holidays = 0;

        /* Copy the private normalized weekmask/holidays data */
        holidays = busdaycal->holidays;
        busdays_in_weekmask = busdaycal->busdays_in_weekmask;
        memcpy(weekmask, busdaycal->weekmask, 7);
        Py_DECREF(busdaycal);
    }
    else {
        /*
         * Fix up the weekmask from the uninitialized
         * signal value to a proper default.
         */
        if (weekmask[0] == 2) {
            weekmask[0] = 1;
        }

        /* Count the number of business days in a week */
        busdays_in_weekmask = 0;
        for (i = 0; i < 7; ++i) {
            busdays_in_weekmask += weekmask[i];
        }

        /* The holidays list must be normalized before using it */
        normalize_holidays_list(&holidays, weekmask);
    }

    /* Make 'dates' into an array */
    if (HPyArray_Check(ctx, dates_in)) {
        dates = HPy_Dup(ctx, dates_in);
        // Py_INCREF(dates);
    }
    else {
        HPy datetime_dtype; // PyArray_Descr *

        /* Use the datetime dtype with generic units so it fills it in */
        datetime_dtype = HPyArray_DescrFromType(ctx, NPY_DATETIME);
        if (HPy_IsNull(datetime_dtype)) {
            goto fail;
        }

        /* This steals the datetime_dtype reference */
        dates = HPyArray_FromAny(ctx, dates_in,
                                                datetime_dtype,
                                                0, 0, 0, HPy_NULL);
        if (HPy_IsNull(dates)) {
            goto fail;
        }
    }

    /* Make sure 'out' is an array if it's provided */
    if (!HPy_IsNull(out_in)) {
        if (!HPyArray_Check(ctx, out_in)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "busday_offset: must provide a NumPy array for 'out'");
            goto fail;
        }
        out = out_in;
    }

    ret = is_business_day(ctx, dates, out,
                    weekmask, busdays_in_weekmask,
                    holidays.begin, holidays.end);

    HPy_Close(ctx, dates);
    if (allocated_holidays && holidays.begin != NULL) {
        PyArray_free(holidays.begin);
    }

    if (HPy_IsNull(out)) {
        out = HPyArray_Return(ctx, ret);
        HPy_Close(ctx, ret);
        ret = out;
    }
    HPyTracker_Close(ctx, ht);
    return ret;

fail:
    HPyTracker_Close(ctx, ht);
    HPy_Close(ctx, dates);
    if (allocated_holidays && holidays.begin != NULL) {
        PyArray_free(holidays.begin);
    }

    return HPy_NULL;
}
