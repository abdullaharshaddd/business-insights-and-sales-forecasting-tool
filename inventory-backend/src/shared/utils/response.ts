import { Response } from 'express';
import { PAGINATION } from '../../config/constants';

export interface PaginationParams {
  page: number;
  limit: number;
}

export interface PaginatedResult<T> {
  success: true;
  data: T[];
  meta: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

export function parsePagination(query: Record<string, unknown>): PaginationParams {
  const page = Math.max(1, parseInt(String(query.page || PAGINATION.DEFAULT_PAGE), 10));
  const limit = Math.min(
    PAGINATION.MAX_LIMIT,
    Math.max(1, parseInt(String(query.limit || PAGINATION.DEFAULT_LIMIT), 10))
  );
  return { page, limit };
}

export function paginatedResponse<T>(
  data: T[],
  total: number,
  page: number,
  limit: number
): PaginatedResult<T> {
  return {
    success: true,
    data,
    meta: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit),
    },
  };
}

export function successResponse<T>(data: T) {
  return { success: true as const, data };
}

export function sendSuccess(res: Response, data: unknown, statusCode = 200) {
  return res.status(statusCode).json(successResponse(data));
}

export function sendPaginated<T>(
  res: Response,
  data: T[],
  total: number,
  page: number,
  limit: number
) {
  return res.status(200).json(paginatedResponse(data, total, page, limit));
}
