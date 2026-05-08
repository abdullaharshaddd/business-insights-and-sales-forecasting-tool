import prisma from '../../config/database';
import { CreateCategoryInput, UpdateCategoryInput } from './category.schema';
import { NotFoundError, ConflictError } from '../../shared/errors/AppError';

export class CategoryService {
  async findAll() {
    const categories = await prisma.category.findMany({
      where: { isActive: true },
      include: {
        children: { where: { isActive: true }, orderBy: { sortOrder: 'asc' } },
        _count: { select: { products: true } },
      },
      orderBy: { sortOrder: 'asc' },
    });

    // Build tree: return only root categories (parentId is null) with nested children
    const roots = categories.filter((c) => c.parentId === null);
    return roots;
  }

  async findById(id: string) {
    const category = await prisma.category.findUnique({
      where: { id },
      include: {
        children: { where: { isActive: true }, orderBy: { sortOrder: 'asc' } },
        parent: true,
        _count: { select: { products: true } },
      },
    });
    if (!category) throw new NotFoundError('Category', id);
    return category;
  }

  async create(input: CreateCategoryInput) {
    const existing = await prisma.category.findUnique({ where: { slug: input.slug } });
    if (existing) throw new ConflictError(`Category with slug '${input.slug}' already exists`);

    if (input.parentId) {
      const parent = await prisma.category.findUnique({ where: { id: input.parentId } });
      if (!parent) throw new NotFoundError('Parent category', input.parentId);
    }

    return prisma.category.create({
      data: input,
      include: { parent: true },
    });
  }

  async update(id: string, input: UpdateCategoryInput) {
    await this.findById(id);

    if (input.slug) {
      const existing = await prisma.category.findFirst({
        where: { slug: input.slug, NOT: { id } },
      });
      if (existing) throw new ConflictError(`Category with slug '${input.slug}' already exists`);
    }

    return prisma.category.update({
      where: { id },
      data: input,
      include: { parent: true, _count: { select: { products: true } } },
    });
  }

  async softDelete(id: string) {
    await this.findById(id);
    return prisma.category.update({
      where: { id },
      data: { isActive: false },
    });
  }
}

export const categoryService = new CategoryService();
