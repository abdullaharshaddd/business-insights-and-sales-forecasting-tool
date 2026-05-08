import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import prisma from '../../config/database';
import { env } from '../../config/env';
import { RegisterInput, LoginInput } from './auth.schema';
import { ConflictError, UnauthorizedError } from '../../shared/errors/AppError';
import { AuthPayload } from '../../middleware/auth.middleware';

export class AuthService {
  async register(input: RegisterInput) {
    const existing = await prisma.user.findUnique({ where: { email: input.email } });
    if (existing) {
      throw new ConflictError(`User with email '${input.email}' already exists`);
    }

    const passwordHash = await bcrypt.hash(input.password, 12);
    const user = await prisma.user.create({
      data: {
        email: input.email,
        passwordHash,
        fullName: input.fullName,
        role: input.role || 'staff',
      },
      select: { id: true, email: true, fullName: true, role: true, createdAt: true },
    });

    return user;
  }

  async login(input: LoginInput) {
    const user = await prisma.user.findUnique({ where: { email: input.email } });
    if (!user || !user.isActive) {
      throw new UnauthorizedError('Invalid email or password');
    }

    const valid = await bcrypt.compare(input.password, user.passwordHash);
    if (!valid) {
      throw new UnauthorizedError('Invalid email or password');
    }

    const payload: AuthPayload = { userId: user.id, email: user.email, role: user.role };
    const accessToken = jwt.sign(payload, env.JWT_SECRET, { expiresIn: env.JWT_EXPIRES_IN as any });
    const refreshToken = jwt.sign(payload, env.JWT_REFRESH_SECRET, { expiresIn: env.JWT_REFRESH_EXPIRES_IN as any });

    return {
      accessToken,
      refreshToken,
      user: { id: user.id, email: user.email, fullName: user.fullName, role: user.role },
    };
  }

  async refreshToken(token: string) {
    try {
      const decoded = jwt.verify(token, env.JWT_REFRESH_SECRET) as AuthPayload;
      const user = await prisma.user.findUnique({ where: { id: decoded.userId } });
      if (!user || !user.isActive) {
        throw new UnauthorizedError('User not found or inactive');
      }

      const payload: AuthPayload = { userId: user.id, email: user.email, role: user.role };
      const accessToken = jwt.sign(payload, env.JWT_SECRET, { expiresIn: env.JWT_EXPIRES_IN as any });
      return { accessToken };
    } catch {
      throw new UnauthorizedError('Invalid refresh token');
    }
  }

  async getProfile(userId: string) {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { id: true, email: true, fullName: true, role: true, isActive: true, createdAt: true },
    });
    if (!user) throw new UnauthorizedError('User not found');
    return user;
  }
}

export const authService = new AuthService();
