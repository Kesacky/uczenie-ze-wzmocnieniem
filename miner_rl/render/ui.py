"""Menu i podsumowania rysowane również bez okna."""
from dataclasses import dataclass
import pygame

TITLES = {'exit': 'Wyszedłeś z kopalni!', 'collapse': 'Zawał!',
          'out_of_oil': 'Zgasła lampa', 'step_limit': 'Limit kroków'}


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    key: str


def hit_test(buttons, pos):
    return next((b.key for b in buttons if b.rect.collidepoint(pos)), None)


def summary(env, record=None, new_record=False):
    info = env.last_info
    return dict(reason=info.get('reason'), score=env.total_reward,
                carried=info.get('collected_value', 0) if info.get('reason') == 'exit' else 0,
                lost=info.get('lost_value', 0), collected=info.get('collected', 0),
                deposits=env.n_deposits, steps=info.get('steps', env.steps),
                record=env.total_reward if record is None else record, new_record=new_record)


def make_buttons(surface, choices, start=0.55, end=0.94):
    w, h = surface.get_size()
    gap = max(2, round(h*0.009))
    height = max(1, int(h*(end-start)/max(1, len(choices)))-gap)
    return [Button(pygame.Rect(round(w*.08), round(h*start)+i*(height+gap),
                               round(w*.84), height), label, key)
            for i, (key, label) in enumerate(choices)]


def _font(size):
    pygame.font.init()
    return pygame.font.Font(pygame.font.match_font('dejavusans'), max(8, round(size)))


def _text(surface, text, rect, color=(244, 229, 195), size=22):
    font = _font(size)
    rendered = font.render(text, True, color)
    if rendered.get_width() > rect.width:
        factor = rect.width/rendered.get_width()
        rendered = pygame.transform.smoothscale(rendered, (rect.width, max(1, round(rendered.get_height()*factor))))
    surface.blit(rendered, rendered.get_rect(center=rect.center))


def _buttons(surface, buttons, mouse_pos, selected):
    for i, button in enumerate(buttons):
        active = button.rect.collidepoint(mouse_pos) or i == selected
        pygame.draw.rect(surface, (112, 81, 42) if active else (57, 48, 39), button.rect, border_radius=5)
        pygame.draw.rect(surface, (216, 170, 78), button.rect, width=1, border_radius=5)
        _text(surface, button.label, button.rect.inflate(-12, -2), size=min(24, button.rect.height*.5))


def draw_menu(surface, buttons, title='Górnik w niestabilnej kopalni', mouse_pos=(-1, -1), selected=None):
    surface.fill((23, 20, 19))
    w, h = surface.get_size()
    for y in range(0, h, 32):
        for x in range(0, w, 32):
            pygame.draw.rect(surface, (35+(x//32+y//32)%3*3, 30, 25), (x+1, y+1, 30, 30))
    _text(surface, title, pygame.Rect(w*.05, h*.06, w*.9, h*.15), size=h*.05)
    _text(surface, 'Strzałki / 1–9 i Enter • Esc: wstecz', pygame.Rect(w*.05, h*.23, w*.9, h*.06), size=h*.025)
    _buttons(surface, buttons, mouse_pos, selected)
    return surface


def draw_end_screen(surface, summary, buttons=(), fonts=None, mouse_pos=(-1, -1), selected=None):
    w, h = surface.get_size()
    shade = pygame.Surface((w, h), pygame.SRCALPHA)
    shade.fill((0, 0, 0, 175))
    surface.blit(shade, (0, 0))
    pygame.draw.rect(surface, (36, 30, 25), (w*.04, h*.035, w*.92, h*.93), border_radius=8)
    _text(surface, TITLES.get(summary['reason'], 'Podsumowanie epizodu'),
          pygame.Rect(w*.08, h*.06, w*.84, h*.09), size=h*.045)
    lines = [f'Wynik: {summary["score"]:.1f}', f'Wyniesiona wartość ładunku: {summary["carried"]}',
             f'Utracony ładunek: {summary["lost"]}',
             f'Wydobyte złoża: {summary["collected"]}/{summary["deposits"]}',
             f'Kroki: {summary["steps"]}', f'Rekord: {summary["record"]:.1f}' +
             (' — Nowy rekord!' if summary['new_record'] else '')]
    for i, line in enumerate(lines):
        _text(surface, line, pygame.Rect(w*.08, h*(.17+i*.052), w*.84, h*.05), size=h*.029)
    _buttons(surface, buttons, mouse_pos, selected)
    return surface
