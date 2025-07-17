#include <raylib.h>
#include "game.h"
#include "colors.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

const int screenWidth = 500;
const int screenHeight = 700;
const int cellSize = 30;

double lastUpdateTime = 0;

bool EventTriggered(double interval)
{
    double currentTime = GetTime() / 3;
    if (currentTime - lastUpdateTime >= interval)
    {
        lastUpdateTime = currentTime;
        return true;
    }
    return false;
}

// Estrutura para peça animada no menu
struct MenuPiece {
    Vector2 position;
    Color color;
};

std::vector<MenuPiece> fallingPieces;

void InitMenuPieces() {
    for (int i = 0; i < 20; i++) {
        MenuPiece p;
        p.position = {
            static_cast<float>(rand() % (screenWidth / cellSize) * cellSize),
            static_cast<float>(-rand() % screenHeight)
        };
        p.color = (Color){ 
            (unsigned char)(100 + rand() % 155),
            (unsigned char)(100 + rand() % 155),
            (unsigned char)(100 + rand() % 155),
            255 
        };
        fallingPieces.push_back(p);
    }
}

void UpdateMenuPieces() {
    for (auto &piece : fallingPieces) {
        piece.position.y += 1;
        if (piece.position.y > screenHeight) {
            piece.position.y = -cellSize * 4;
            piece.position.x = rand() % (screenWidth / cellSize) * cellSize;
        }
    }
}

void DrawMenuPieces() {
    for (auto &piece : fallingPieces) {
        DrawRectangle(piece.position.x, piece.position.y, cellSize, cellSize, piece.color);
        DrawRectangleLines(piece.position.x, piece.position.y, cellSize, cellSize, Fade(WHITE, 0.4f));
    }
}

void DrawGameOverScreen(Font font, bool hoverRestart, bool hoverExit) {
    DrawRectangle(0, 0, screenWidth, screenHeight, Fade(BLACK, 0.85f));
    DrawTextEx(font, "GAME OVER", {screenWidth / 2 - MeasureTextEx(font, "GAME OVER", 60, 2).x / 2, 100}, 60, 2, RED);

    Color restartColor = hoverRestart ? SKYBLUE : DARKGRAY;
    Color exitColor = hoverExit ? SKYBLUE : DARKGRAY;

    Rectangle restartBtn = {screenWidth / 2 - 100, 220, 200, 50};
    Rectangle exitBtn = {screenWidth / 2 - 100, 290, 200, 50};

    DrawRectangleRounded(restartBtn, 0.3, 10, restartColor);
    DrawRectangleRounded(exitBtn, 0.3, 10, exitColor);

    DrawTextEx(font, "Reiniciar", {restartBtn.x + 35, restartBtn.y + 10}, 30, 2, WHITE);
    DrawTextEx(font, "Sair", {exitBtn.x + 75, exitBtn.y + 10}, 30, 2, WHITE);
}

int main()
{
    InitWindow(screenWidth, screenHeight, "Tetris Computacional Vision");
    SetWindowPosition(850, 50);
    SetTargetFPS(60);

    Font font = LoadFontEx("monogram.ttf", 64, 0, 0);

    srand(time(nullptr));
    InitMenuPieces();

    Game game;

    while (!WindowShouldClose())
    {
        if (game.inMenu) {
            BeginDrawing();
            ClearBackground({32, 32, 80});

            UpdateMenuPieces();
            DrawMenuPieces();

            DrawText("TETRIS", screenWidth / 2 - MeasureText("TETRIS", 60) / 2, 60, 60, RAYWHITE);
            const char *desc = "Controle com visao computacional";
            DrawText(desc, screenWidth / 2 - MeasureText(desc, 20) / 2, 130, 20, GRAY);

            Rectangle startArea = {screenWidth / 2 - 160, 190, 320, 50};
            DrawRectangleRounded(startArea, 0.3f, 10, BLUE);
            DrawText("     ROTACIONE PARA INICIAR", startArea.x + 10, startArea.y + 15, 18, WHITE);

            Rectangle exitBtn = {screenWidth / 2 - 75, 260, 150, 40};
            DrawRectangleRounded(exitBtn, 0.3f, 10, DARKGRAY);
            DrawText("EXIT", exitBtn.x + 45, exitBtn.y + 10, 20, WHITE);

            Vector2 mouse = GetMousePosition();
            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && CheckCollisionPointRec(mouse, exitBtn)) {
                CloseWindow();
            }

            game.HandleInput();
            EndDrawing();
        } 
        else  {
            BeginDrawing();
            ClearBackground(darkBlue);

            if (!game.gameOver) {
                UpdateMusicStream(game.music);
                game.HandleInput();
                if (EventTriggered(0.2)) {
                    game.MoveBlockDown();
                }
            }

            DrawTextEx(font, "Score", {365, 15}, 38, 2, WHITE);
            DrawTextEx(font, "Next", {370, 175}, 38, 2, WHITE);

            DrawRectangleRounded({320, 55, 170, 60}, 0.3, 6, lightBlue);
            char scoreText[10];
            sprintf(scoreText, "%d", game.score);
            Vector2 textSize = MeasureTextEx(font, scoreText, 38, 2);
            DrawTextEx(font, scoreText, {320 + (170 - textSize.x) / 2, 65}, 38, 2, WHITE);

            DrawRectangleRounded({320, 215, 170, 180}, 0.3, 6, lightBlue);
            game.Draw();

            DrawTextEx(font, ("COMANDO: " + game.lastCommand).c_str(), {30, 640}, 20, 2, WHITE);

            if (game.gameOver) {
                Vector2 mouse = GetMousePosition();
                bool hoverRestart = CheckCollisionPointRec(mouse, {screenWidth / 2 - 100, 220, 200, 50});
                bool hoverExit = CheckCollisionPointRec(mouse, {screenWidth / 2 - 100, 290, 200, 50});

                DrawGameOverScreen(font, hoverRestart, hoverExit);

                if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                    if (hoverRestart) {
                        game.Reset();
                        game.inMenu = true;
                    }
                    if (hoverExit) {
                        CloseWindow();
                    }
                }
            }

            EndDrawing();
        }
    }

    CloseWindow();
    return 0;
}
